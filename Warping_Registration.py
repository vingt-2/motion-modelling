import numpy as np
from scipy import signal, interpolate
import pyqtgraph as pg
import dtw

GAUSS_KERNEL = np.loadtxt("gauss_kernel.csv", delimiter=",")

def _static_differentiate_trajectory(trajectory, order, smooth=False):

	T = trajectory
	dimension = len(T[0])
	derivative = np.zeros((len(T), dimension))
	if order == 1:
		for i in range(1, len(T)):
		#derivative[i] = (-T[i + 2] + 8 * T[i + 1] - 8 * T[i - 1] + T[i - 2]) / 12.0
			derivative[i] = T[i] - T[i - 1]
	elif order == 2:
		for i in range(1, len(T) - 1):
		#derivative[i] = (-T[i + 2] + 16 * T[i + 1] - 30 * T[i] + 16 * T[i - 1] - T[i - 2]) / 12.0
			derivative[i] = (T[i - 1] - 2 * T[i] + T[i + 1])
	else:
		print("(Static_diff) Derivative of order: " + str(order) + " not supported.")

	if smooth:
		unpacked_traj = np.transpose(derivative)

		convolved_dimensions = []
		for dim_traj in unpacked_traj:
			convolve_dim_traj = np.convolve(dim_traj, GAUSS_KERNEL)
			convolved_dimensions.append(convolve_dim_traj)

		derivative = np.transpose(convolved_dimensions)[:len(T)]

	return derivative

def _static_warp_motion(warp, motion, interpolate=True):
	warped_motion = np.zeros(len(warp))

	if interpolate:
		for i in range(0, len(warp)):
			t = warp[i]
			f = int(np.floor(t))
			c = min(int(np.ceil(t)), len(motion) - 1)
			vf = motion[f]
			vc = motion[c]
			warped_motion[i] = float(np.interp([t], [f, c], [vf, vc]))
	else:
		warped_motion = np.array([motion[int(t)] for t in warp])

	return warped_motion

class GreedyWarp(object):
	def __init__(self, ref_traj, window_size):

		self.evaluated_warp = [0, 0]
		self.reconstructed_motion = []

		self.motion_dimension = len(ref_traj[0])

		self.ref_traj = ref_traj
		self.input_traj = []
		self.window_size = window_size

		self.input_feature_list = []
		self.ref_feature_list = []

		self.pos_weight = 1
		self.vel_weight = 1

		self._prepare_ref_features(True)

	def _start_zero(self, list):
		list = [l - list[0] for l in list]

		return list

	def _prepare_ref_features(self, normalize=False):

		butter_b, butter_a = signal.butter(2, 0.1, 'low', analog=False)

		ref_pos = signal.lfilter(butter_b, butter_a, self.ref_traj, axis=0)
		ref_vel = _static_differentiate_trajectory(ref_pos, order=1, smooth=False)
		ref_acc = _static_differentiate_trajectory(ref_pos, order=2, smooth=False)

		ref_pos_norm = self._start_zero(ref_pos)
		ref_vel_norm = self._start_zero(ref_vel)
		ref_acc_norm = self._start_zero(ref_acc)

		self.ref_feature_list = []
		for i in range(len(ref_pos)):
			feature_at_i = map(lambda x:self.pos_weight*x,ref_pos[i]) + map(lambda x:self.vel_weight*x,ref_vel[i])
			self.ref_feature_list += [feature_at_i]

	def update_warp(self, new_pos):
		self.input_traj += [new_pos]

		butter_b, butter_a = signal.butter(2, 0.1, 'low', analog=False)

		input_so_far_filtered = signal.lfilter(butter_b, butter_a, self.input_traj, axis=0)

		new_pos = np.array(input_so_far_filtered[-1])

		if len(self.input_feature_list) < 1:
			new_vel = np.zeros(self.motion_dimension)
		else:
			old_pos = self.input_feature_list[-1][:self.motion_dimension]
			new_vel = new_pos - old_pos

		new_input_feature = [] + (self.pos_weight*new_pos).tolist() + (self.vel_weight*new_vel).tolist()

		self.input_feature_list += [new_input_feature]

		best_match = (np.float('inf'), -1)
		a = self.evaluated_warp[-1] + 1
		b = min(a + self.window_size, len(self.ref_feature_list) - 1)
		tail_window = 2
		for j in range(a, b):
			input_data_point = []
			ref_data_point = []
			for tw in range(0, tail_window):
				weight = np.exp(-float(tw) / tail_window)
				ref_data_point += map(lambda x:x*weight, self.ref_feature_list[j - tw])
				input_data_point += map(lambda x:x*weight, self.input_feature_list[-tw])

			difference = np.array(ref_data_point) - np.array(input_data_point)

			objective_function = np.linalg.norm(difference)

			if objective_function < best_match[0]:
				best_match = (objective_function, j)

		time_match = best_match[1]

		if time_match < self.evaluated_warp[-1]:
			time_match = self.evaluated_warp[-1]

		self.evaluated_warp.append(time_match)

	def warp_motion(self, motion, interpolate=True):
		warped_motion = np.zeros(len(self.evaluated_warp))

		if interpolate:
			for i in range(0, len(self.evaluated_warp)):
				t = self.evaluated_warp[i]
				f = int(np.floor(t))
				c = min(int(np.ceil(t)), len(motion) - 1)
				vf = motion[f]
				vc = motion[c]
				warped_motion[i] = float(np.interp([t], [f, c], [vf, vc]))
		else:
			warped_motion = np.array([motion[int(t)] for t in self.evaluated_warp])

		return warped_motion

	def invert_warp(self, warp, motion_range=()):
		inted_warp = [int(w) for w in warp]

		(start, end) = (inted_warp[0], inted_warp[-1])  # from monotic property

		if motion_range != ():
			(start, end) = (max(start, motion_range[0]), min(end, motion_range[1]))

		inv_warp = np.zeros(end - start)
		interpolate = []
		for i in range(start, end):
			try:
				inv_warp_v = inted_warp.index(i)
				inv_warp[i - start] = inv_warp_v
			except ValueError:
				# NOT VERY SMOOTH, be careful
				interpolate.append(i - start)
		for i in interpolate:
			before = inv_warp[i - 1]
			after_indx = i + 1
			while after_indx in interpolate:
				after_indx += 1
			if after_indx == len(inv_warp):
				inv_warp[i] = inv_warp[i - 1]
			else:
				after = inv_warp[after_indx]
				inv_warp[i] = float(np.interp([i], [i - 1, after_indx], [before, after]))

		self.evaluated_warp = inv_warp

class OnlineDynamicTimeWarp(object):
	def __init__(self, ref_trajectory, distance_func=lambda x, y: np.linalg.norm(x - y), subtract_offset=True):
		self.evaluated_warp = []
		self.ref_trajectory = ref_trajectory
		self.subtract_offset = subtract_offset
		self.motion_dimension = len(self.ref_trajectory[0])
		self.input_traj_so_far = []
		self.input_offset = np.zeros(self.motion_dimension) #!!! Or Replace with dimension of data
		self.input_feature_list = []
		self.ref_feature_list = []

		self.distance_function = distance_func

		self.vel_importance_weight = 1
		self.pos_importance_weight = 1

		self._prepare_ref_features()

		self.distance_matrix = np.zeros((1, len(self.ref_feature_list) + 1))

		self.distance_matrix[0, 1:] = np.inf
		self.distance_matrix[1:, 0] = np.inf
		self.cost_matrix = self.distance_matrix.copy()

	def _prepare_ref_features(self):

		butter_b, butter_a = signal.butter(2, 0.1, 'low', analog=False)

		ref_pos = signal.lfilter(butter_b, butter_a, self.ref_trajectory, axis=0)
		ref_vel = _static_differentiate_trajectory(ref_pos, order=1, smooth=False)
		ref_acc = _static_differentiate_trajectory(ref_pos, order=2, smooth=False)

		ref_pos_norm = [l - ref_pos[0] for l in ref_pos] if self.subtract_offset else ref_pos

		ref_features = ([self.pos_importance_weight * a for a in ref_pos_norm],
		                [self.vel_importance_weight * a for a in ref_vel])

		self.ref_feature_list = np.array([np.array(f).flatten() for f in zip(*ref_features)])

		# p = pg.plot(title="ref_pos")
		# p.plot([x[0] for x in ref_pos])

		self.input_feature_list = np.array([])

	def _update_input_features(self, new_input_point):
		self.input_traj_so_far += [new_input_point]

		butter_b, butter_a = signal.butter(2, 0.1, 'low', analog=False)

		input_so_far_filtered = signal.lfilter(butter_b, butter_a, self.input_traj_so_far, axis=0)

		new_input_point = input_so_far_filtered[-1]

		if len(self.input_feature_list) < 1:
			new_vel, new_point = np.zeros(self.motion_dimension), new_input_point
			self.input_offset = new_input_point if self.subtract_offset else np.zeros(self.motion_dimension)
		else:
			new_point = new_input_point - self.input_offset
			new_vel = new_point - self.input_feature_list[-1][:self.motion_dimension] # !!!!! OR replace 3 with dimension of data

		new_feature = np.array([self.pos_importance_weight*new_point, self.vel_importance_weight*new_vel]).flatten()

		features = np.zeros((len(self.input_feature_list)+1, 2*self.motion_dimension)) # !!!!! OR replace 3 with dimension of data
		if len(self.input_feature_list) > 0:
			features[:-1, :] = self.input_feature_list
			features[-1] = new_feature
			self.input_feature_list = features
		else:
			self.input_feature_list = np.array([new_feature])

	def _update_dtw(self, new_input_point):
		self._update_input_features(new_input_point)

		#Update cost matrix with the cost to the new positions
		new_D_row = np.zeros(len(self.ref_feature_list)+1)
		new_D_row[0] = np.inf
		for j in range(1,len(new_D_row)):
			new_D_row[j] = self.distance_function(self.input_feature_list[-1], self.ref_feature_list[j-1])

		new_dist_mat = np.zeros((self.distance_matrix.shape[0] + 1, self.distance_matrix.shape[1]))
		new_dist_mat[:-1,:] = self.distance_matrix
		new_dist_mat[-1] = np.array(new_D_row)

		self.distance_matrix = new_dist_mat

		new_acc_row = new_D_row.copy()
		if self.cost_matrix.shape[0] > 1:
			for j in range(1, len(new_D_row)):
				new_acc_row[j] += min(self.cost_matrix[-1, j - 1], self.cost_matrix[-1, j], new_acc_row[j - 1])

		new_acc_cost = np.zeros((self.cost_matrix.shape[0] + 1, self.cost_matrix.shape[1]))
		new_acc_cost[:-1, :] = self.cost_matrix
		new_acc_cost[-1] = new_acc_row

		self.cost_matrix = new_acc_cost

		if len(self.input_feature_list) == 1:
			path = np.zeros(len(self.ref_feature_list)), np.zeros(len(self.ref_feature_list))
		elif len(self.ref_feature_list) == 1:
			path =  np.zeros(len(self.input_feature_list)), np.zeros(len(self.input_feature_list))
		else:
			path = self._traceback()

		return self.cost_matrix[-1, -1] / sum(self.cost_matrix.shape)-2, self.distance_matrix, self.cost_matrix, path

	#To Do: Incorporate Penalty for long jumps in the traceback
	def _traceback(self):
		i, j = np.array(self.cost_matrix.shape) - 2

		last_best_warp = self.evaluated_warp[-1] if len(self.evaluated_warp) > 0 else 0

		# Do NOT traceback from the last position of the reference motion
		# Instead, find which one makes more sense.
		j = np.argmin([np.exp(abs(last_best_warp-p)/100)*self.cost_matrix[i, p] for p in range(0, j)])
		#j = np.argmin([self.cost_matrix[i, p] for p in range(0, j)])
		p, q = [i], [j]
		while ((i > 0) or (j > 0)):
			tb = np.argmin((self.cost_matrix[i, j], self.cost_matrix[i, j + 1], self.cost_matrix[i + 1, j]))
			if (tb == 0):
				i -= 1
				j -= 1
			elif (tb == 1):
				i -= 1
			else:  # (tb == 2):
				j -= 1
			p.insert(0, i)
			q.insert(0, j)
		return np.array(p), np.array(q)

	def update_warp(self, new_point):
		dist, cost, acc, path = self._update_dtw(new_point)

		path_t = np.transpose(path).tolist()

		warp = np.ones(len(self.input_feature_list))

		for p in path_t:
			if p[0] < len(warp):
				index = int(np.floor(p[0]))
				warp[index] = p[1]

		self.evaluated_warp = warp

		if len(self.evaluated_warp) == 0:
			self.evaluated_warp = [0]

		return True

	def warp_motion(self, motion, interpolate=True):
		warped_motion = np.zeros(len(self.evaluated_warp))

		if interpolate:
			for i in range(0, len(self.evaluated_warp)):
				t = self.evaluated_warp[i]
				f = int(np.floor(t))
				c = min(int(np.ceil(t)), len(motion) - 1)
				vf = motion[f]
				vc = motion[c]
				warped_motion[i] = float(np.interp([t], [f, c], [vf, vc]))
		else:
			warped_motion = np.array([motion[int(t)] for t in self.evaluated_warp])

		return warped_motion

	def invert_warp(self, warp, motion_range=()):
		inted_warp = [int(w) for w in warp]

		(start, end) = (inted_warp[0], inted_warp[-1])  # from monotic property

		if motion_range != ():
			(start, end) = (max(start, motion_range[0]), min(end, motion_range[1]))

		inv_warp = np.zeros(end - start)
		interpolate = []
		for i in range(start, end):
			try:
				inv_warp_v = inted_warp.index(i)
				inv_warp[i - start] = inv_warp_v
			except ValueError:
				# NOT VERY SMOOTH, be careful
				interpolate.append(i - start)
		for i in interpolate:
			before = inv_warp[i - 1]
			after_indx = i + 1
			while after_indx in interpolate:
				after_indx += 1
			if after_indx == len(inv_warp):
				inv_warp[i] = inv_warp[i - 1]
			else:
				after = inv_warp[after_indx]
				inv_warp[i] = float(np.interp([i], [i - 1, after_indx], [before, after]))

		self.evaluated_warp = inv_warp

class SaliencyTimeWarp(object):

	def __init__(self, ref_traj, show_graphs=False):

		self.show_graphs = show_graphs

		self.evaluated_warp = [0]
		self.motion_dimension = len(ref_traj[0])

		self.ref_traj = ref_traj
		self.input_traj = []

		self.butter_b, self.butter_a = signal.butter(2, 0.1, 'low', analog=False)

		first_p = ref_traj[0]
		to_filter_list = ref_traj.tolist()
		for i in range(100):
			to_filter_list.insert(0,first_p)

		self.filtered_ref_pos = signal.lfilter(self.butter_b, self.butter_a, to_filter_list, axis=0)[100:]

		#self.filtered_ref_pos = self.filtered_ref_pos[30:-20]

		self.filtered_ref_vel = _static_differentiate_trajectory(self.filtered_ref_pos, order=1, smooth=False)
		self.filtered_ref_acc = _static_differentiate_trajectory(self.filtered_ref_pos, order=2, smooth=False)


		filtered_acc_tp = np.transpose(self.filtered_ref_acc)
		filtered_acc_tp = [filtered_acc_tp_i / max([abs(a) for a in filtered_acc_tp_i]) for filtered_acc_tp_i in filtered_acc_tp]
		self.filtered_ref_acc = np.transpose(filtered_acc_tp)

		self.salient_points_ref = []

		EPSILON = 0.01
		# Sign of the velocity for each PCA component
		vel_signs = [np.sign(vel) for vel in self.filtered_ref_vel[0]]
		acc_signs = [np.sign(acc) for acc in self.filtered_ref_acc[0]]

		for i in range(len(self.filtered_ref_vel)):
			for pca_dim in range(len(self.filtered_ref_vel[i])):
				pos = self.filtered_ref_pos[i][pca_dim]
				vel = self.filtered_ref_vel[i][pca_dim]
				acc = self.filtered_ref_acc[i][pca_dim]

				if (np.sign(vel) != vel_signs[pca_dim] and vel_signs[pca_dim] != 0.0)  and abs(acc) > EPSILON:
					self.salient_points_ref += [(pca_dim, i-1, pos)]

				elif np.sign(acc) != acc_signs[pca_dim] and acc_signs[pca_dim] != 0.0:
					self.salient_points_ref += [(pca_dim, i - 1, pos)]

				vel_signs[pca_dim] = np.sign(vel)
				acc_signs[pca_dim] = np.sign(acc)

		if self.show_graphs :
			self.ref_saliency_plot = pg.plot(title="Ref Saliency")
			self.input_saliency_plot = pg.plot(title="Input Saliency")

			self.ref_saliency_plot.addLegend()
			for pca_dim in range(len(self.filtered_ref_pos[0])):
				current_dim_pos = [x[pca_dim] for x in self.filtered_ref_pos]
				self.ref_saliency_plot.plot(current_dim_pos, pen=(pca_dim, len(self.filtered_ref_pos[0])), name="PC# " + str(pca_dim))
			# for (pca_dim, i, pca_pos) in self.salient_points_ref:
			# 	self.ref_saliency_plot.addLine(i, pen=(pca_dim, len(self.filtered_ref_pos[0])))

		self.input_vel_signs = np.zeros(self.motion_dimension)
		self.input_acc_signs = np.zeros(self.motion_dimension)

		self.salient_points_input = []

		self.match_list = [(0,0)]

		self.scores = dict()

		self.minmax_dim = [abs(np.min(pca_dim)-np.max(pca_dim)) for pca_dim in np.transpose(self.filtered_ref_pos)]

		self.ref_plot_progress_line = None

	def update_warp(self, new_pos):

		self.input_traj += [new_pos]

		self.butter_b, self.butter_a = signal.butter(2, 0.1, 'low', analog=False)

		filtered_pos = signal.lfilter(self.butter_b, self.butter_a, self.input_traj, axis=0)

		filtered_vel = _static_differentiate_trajectory(filtered_pos, order=1)
		filtered_acc = _static_differentiate_trajectory(filtered_pos, order=2)

		filtered_acc_tp = np.transpose(filtered_acc)
		filtered_acc_tp = [filtered_acc_tp_i / max([abs(a) for a in filtered_acc_tp_i]) for filtered_acc_tp_i in
		                   filtered_acc_tp]
		filtered_acc = np.transpose(filtered_acc_tp)

		EPSILON = 0.01

		salient_points_input = []

		for pca_dim in range(0, self.motion_dimension):
			pos = filtered_pos[-1][pca_dim]
			vel = filtered_vel[-1][pca_dim]
			acc = filtered_acc[-2][pca_dim] if len(filtered_acc) > 1 else 0

			if (np.sign(vel) != self.input_vel_signs[pca_dim] and self.input_vel_signs[pca_dim] != 0.0) and abs(acc) > EPSILON:
				salient_points_input += [(pca_dim, len(self.input_traj), pos)]
			elif np.sign(acc) != self.input_acc_signs[pca_dim] and self.input_acc_signs[pca_dim] != 0.0:
				salient_points_input += [(pca_dim, len(self.input_traj), pos)]

			self.input_vel_signs[pca_dim] = np.sign(vel)
			self.input_acc_signs[pca_dim] = np.sign(acc)

		if self.show_graphs :
			self.input_saliency_plot.clear()
			for pca_dim in range(len(filtered_pos[0])):
				current_dim_pos = [x[pca_dim] for x in filtered_pos]
				self.input_saliency_plot.plot(current_dim_pos, pen=(pca_dim, len(filtered_pos[0])), name="PC# " + str(pca_dim))
			for (pca_dim, i, pca_pos) in self.salient_points_input:
				self.input_saliency_plot.addLine(i, pen=(pca_dim, len(filtered_pos[0])))

		self.salient_points_input += salient_points_input

		if len(salient_points_input) < 1 or len(self.salient_points_input) < len(filtered_pos[0]): # len(filtered_pos[0]) gives the number of PCA Dimensions
			self.evaluated_warp += [self.evaluated_warp[-1]]
			return

		# score_func = lambda (i_dim, i_t, i_pos), (r_dim, r_t, r_pos): (1 - (
		# abs(i_pos - r_pos) / self.minmax_dim[r_dim]) if i_dim == r_dim else -1) - abs(i_t - r_t) / float(
		# 	len(self.ref_traj))

		score_func = lambda input_p, reg_p: (1 - (
			abs(input_p[2] - reg_p[2]) / self.minmax_dim[reg_p[0]]) if input_p[0] == reg_p[0] else -1) - abs(input_p[1] - reg_p[1]) / float(
			len(self.ref_traj))

		#score_func = lambda (i_dim, i_t, i_pos), (r_dim, r_t, r_pos): 1 if i_dim == r_dim else -1

		def _compute_score(ref_i,in_i):
			try:
				score = self.scores[(ref_i,in_i)]
			except KeyError:
				if ref_i < 0 or in_i < 0:
					return 0

				score = score_func(self.salient_points_input[in_i], self.salient_points_ref[ref_i])
				score += np.max(
					[_compute_score(ref_i - 1, in_i - 1), _compute_score(ref_i - 2, in_i - 1) - 0.5, _compute_score(ref_i - 1, in_i - 2) - 0.5])

				self.scores[(ref_i,in_i)] = score

			return score

		best_pos = -1
		best_score = -np.float("inf")
		for start_ref_i in range(len(self.salient_points_ref)):
			score = _compute_score(start_ref_i, len(self.salient_points_input) - 1)

			if score > best_score:
				best_pos = start_ref_i
				best_score = score

		#print("Input Position= " + str(len(self.input_traj)-1) + " best matches " + str(self.salient_points_ref[best_pos]))

		warping_index = self.salient_points_ref[best_pos][1]

		if warping_index > self.evaluated_warp[-1]:
			self.evaluated_warp += [warping_index]

			if self.ref_plot_progress_line != None:
				self.ref_saliency_plot.removeItem(self.ref_plot_progress_line)

			if self.show_graphs:
				self.ref_plot_progress_line = self.ref_saliency_plot.addLine(warping_index)

			self.match_list += [(len(self.input_traj),self.salient_points_ref[best_pos][1])]
		else:
			self.evaluated_warp += [self.evaluated_warp[-1]]

	def warp_motion(self, motion, interpolate=True):
		warped_motion = np.zeros(len(self.evaluated_warp))

		if interpolate:
			for i in range(0, len(self.evaluated_warp)):
				t = self.evaluated_warp[i]
				f = int(np.floor(t))
				c = min(int(np.ceil(t)), len(motion) - 1)
				vf = motion[f]
				vc = motion[c]
				warped_motion[i] = float(np.interp([t], [f, c], [vf, vc]))
		else:
			warped_motion = np.array([motion[int(t)] for t in self.evaluated_warp])

		return warped_motion

