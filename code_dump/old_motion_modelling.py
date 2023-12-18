import os, time
from utils import qv_mult
import cPickle as pickle
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.mixture import GaussianMixture
from PySide import QtGui, QtCore
import pyqtgraph as pg
from scipy import interpolate, optimize, stats
from NatNet_acquisition import NatNetAcquisitionWidget
from pykalman import KalmanFilter
from dtw import dtw

min_start_time = 0

CAMERA_ID = 1
OBJECT_ID = 2

DB_FILE_EXTENSION = ".modb"

def normal_dist_1d(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def get_blur_trajectory(traj_1, traj_2):
	len_blur_traj = min(len(traj_1), len(traj_2))

	diff = np.abs(np.array(traj_1[:len_blur_traj]) - np.array(traj_2[:len_blur_traj]))
	focus_dist_to_arcmin_coc = 85.3 * 0.5 * np.sqrt(2) * 1.8 * 0.625
	blur_traj = focus_dist_to_arcmin_coc * diff

	return blur_traj

class KFSplineWarp(object):

	def __init__(self):
		self.evaluated_warp = []

	def compute_warp(self, ref_keyframes, keyframes):
		if len(keyframes) != len(ref_keyframes):
			return False

		spline = interpolate.PchipInterpolator(ref_keyframes, keyframes, axis=0)

		self.evaluated_warp = [float(spline(x)) for x in range(ref_keyframes[0], ref_keyframes[-1])]

		return True

	def warp_motion(self, motion, interpolate=True):
		warped_motion = np.zeros(len(self.evaluated_warp))

		if interpolate:
			for i in range(0,len(self.evaluated_warp)):
				t = self.evaluated_warp[i]
				f = int(np.floor(t))
				c = min(int(np.ceil(t)),len(motion)-1)
				vf = motion[f]
				vc = motion[c]
				warped_motion[i] = float(np.interp([t], [f, c], [vf, vc]))
		else:
			warped_motion = np.array([motion[int(t)] for t in self.evaluated_warp])

		return warped_motion

	def invert_warp(self, warp, motion_range=()):
		inted_warp = [int(w) for w in warp]

		(start, end) = (inted_warp[0], inted_warp[-1]) # from monotic property

		if motion_range!=():
			(start, end) = (max(start,motion_range[0]), min(end,motion_range[1]))

		inv_warp = np.zeros(end-start)
		interpolate = []
		for i in range(start,end):
			try:
				inv_warp_v = inted_warp.index(i)
				inv_warp[i - start] = inv_warp_v
			except ValueError:
				# NOT VERY SMOOTH, be careful
				interpolate.append(i-start)
		for i in interpolate:
			before = inv_warp[i-1]
			after_indx = i+1
			while after_indx in interpolate:
				after_indx += 1
			if after_indx == len(inv_warp):
				inv_warp[i] = inv_warp[i-1]
			else:
				after = inv_warp[after_indx]
				inv_warp[i] = float(np.interp([i], [i-1, after_indx], [before, after]))

		self.evaluated_warp = inv_warp

class GreedyWarp(object):
	def __init__(self, reference_motion, input_motion, window_size):
		self.supersample_rate = 4
		self.evaluated_warp = [0,0]
		self.reconstructed_motion = []
		self.match_error = 0
		self.reference_motion = reference_motion
		self.input_motion = input_motion
		self.window_size = window_size

		self.input_feature_list = None

		self.ref_vel = None
		self.ref_feature_list = None
		self.ref_feature_tree = None

	def normalize_list(self, list):
		as_array = np.array(list)
		min_element = min(list)
		as_array = as_array - min_element
		max_element = np.max(np.absolute(as_array),0)
		as_array = (as_array + min_element) / np.max(np.absolute(as_array),0)

		return as_array.tolist()

	def build_ref_features(self, normalize=False):
		ref_ts  = range(0,len(self.reference_motion.trajectory))
		np_pos_arr = np.array(self.reference_motion.trajectory)
		ref_pos = (np_pos_arr - np.average(np_pos_arr)).tolist()
		self.ref_vel = self.reference_motion.differentiate(order=1,smooth=True)
		ref_acc = self.reference_motion.differentiate(order=2,smooth=True)

		in_ts  = range(0,len(self.input_motion.trajectory))
		np_pos_arr = np.array(self.input_motion.trajectory)
		in_pos = (np_pos_arr - np.average(np_pos_arr)).tolist()
		in_vel = self.input_motion.differentiate(order=1,smooth=True)
		in_acc = self.input_motion.differentiate(order=2,smooth=True)

		if normalize:
			ref_pos_norm = self.normalize_list(ref_pos)
			ref_vel_norm = self.normalize_list(self.ref_vel)
			ref_acc_norm = self.normalize_list(ref_acc)

			in_pos_norm = self.normalize_list(in_pos)
			in_vel_norm = self.normalize_list(in_vel)
			in_acc_norm = self.normalize_list(in_acc)

			ref_triple = (ref_pos_norm, [10 * a for a in ref_vel_norm])
			self.ref_feature_list = np.array(zip(*ref_triple))

			in_triple = (in_pos_norm, [10 * a for a in in_vel_norm])
			self.input_feature_list = np.array(zip(*in_triple))
		else:
			ref_triple = (ref_pos, self.ref_vel)
			self.ref_feature_list = np.array(zip(*ref_triple))

			in_triple = (in_pos, in_vel)
			self.input_feature_list = np.array(zip(*in_triple))


		self.ref_feature_tree = KDTree(self.ref_feature_list, leaf_size=15, metric='euclidean')

	def register_next_frame(self):
		if self.ref_feature_list == None:
			self.build_ref_features(True)

		i = len(self.evaluated_warp) - 1

		input_point = self.input_feature_list[i]

		best_match = (np.float('inf'), -1)
		a = self.evaluated_warp[-1]
		b = min(a + self.window_size, len(self.reference_motion.trajectory))
		for j in range(a, b):
			diff = 0
			for m in range(0,5):
				a = np.log(5-m) * (self.ref_feature_list[j-m] - self.input_feature_list[i-m])
				diff += np.dot(a,a)
			distance = np.sqrt(diff)
			objective_function = distance# + 0.1 * (abs(j-i)+1)

			if objective_function < best_match[0]:
				# print objective_function
				# print j
				best_match = (objective_function, j)

		self.evaluated_warp.append(best_match[1])

		self.reconstructed_motion.append(self.reference_motion.trajectory[self.evaluated_warp[-1]])

		self.match_error += (self.reconstructed_motion[-1] - self.input_motion.trajectory[i]) ** 2

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

class DynamicTimeWarp(object):
	def __init__(self):
		self.evaluated_warp = []

	def compute_warp(self, ref_trajectory, input_trajectory):
		dist, cost, acc, path = dtw(ref_trajectory, input_trajectory, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

		self.evaluated_warp = path[1]

		return True

	def warp_motion(self, motion, interpolate=True):
		warped_motion = np.zeros(len(self.evaluated_warp))

		if interpolate:
			for i in range(0,len(self.evaluated_warp)):
				t = self.evaluated_warp[i]
				f = int(np.floor(t))
				c = min(int(np.ceil(t)),len(motion)-1)
				vf = motion[f]
				vc = motion[c]
				warped_motion[i] = float(np.interp([t], [f, c], [vf, vc]))
		else:
			warped_motion = np.array([motion[int(t)] for t in self.evaluated_warp])

		return warped_motion

	def invert_warp(self, warp, motion_range=()):
		inted_warp = [int(w) for w in warp]

		(start, end) = (inted_warp[0], inted_warp[-1]) # from monotic property

		if motion_range!=():
			(start, end) = (max(start,motion_range[0]), min(end,motion_range[1]))

		inv_warp = np.zeros(end - start)
		interpolate = []
		for i in range(start,end):
			try:
				inv_warp_v = inted_warp.index(i)
				inv_warp[i - start] = inv_warp_v
			except ValueError:
				# NOT VERY SMOOTH, be careful
				interpolate.append(i-start)
		for i in interpolate:
			before = inv_warp[i-1]
			after_indx = i+1
			while after_indx in interpolate:
				after_indx += 1
			if after_indx == len(inv_warp):
				inv_warp[i] = inv_warp[i-1]
			else:
				after = inv_warp[after_indx]
				inv_warp[i] = float(np.interp([i], [i-1, after_indx], [before, after]))

		self.evaluated_warp = inv_warp

class Motion(object):

	def __init__(self, name):
		self.print_c = None

		self.name = name
		self.original_trajectory =[]
		self.edited_traj = []
		self.trajectory = []
		self.key_frames = []
		self.warped_motion = []
		self.time_warp = None

	def differentiate(self, order, smooth=False):
		gauss_kernel = np.loadtxt("gauss_kernel.csv", delimiter=",")
		T = self.trajectory
		derivative = np.zeros(len(T)).tolist()
		for i in range(1, len(T) - 1):
			if order == 1:
				#derivative[i] = (-T[i + 2] + 8 * T[i + 1] - 8 * T[i - 1] + T[i - 2]) / 12.0
				derivative[i] = T[i] - T[i - 1]
			elif order == 2:
				#derivative[i] = (-T[i + 2] + 16 * T[i + 1] - 30 * T[i] + 16 * T[i - 1] - T[i - 2]) / 12.0
				derivative[i] = (T[i - 1] - 2 * T[i] + T[i + 1])
			else:
				self.print_c("Derivative of order: " + str(order) + " not supported.")

		if smooth:
			derivative = filtered_y = np.convolve(derivative, gauss_kernel)

		return derivative

class MotionDatabase(object):
	def __init__(self):
		self.print_c = None

		self.dbFilename = ""
		self.reference_motion = "None"
		self.motions = []
		self.motion_model = None

	def add_motion(self, name, motion_traj):
		new_traj = Motion(name)
		new_traj.trajectory = motion_traj
		new_traj.original_trajectory = motion_traj
		new_traj.edited_traj = motion_traj

		self.motions.append(new_traj)

	def load(self, data_path, filenames=None):

		(frame_start, frame_end) = (None, None)  # Not supported for now... well see

		if filenames == None:
			filenames = [f for f in os.listdir(data_path) if (".pickle" in f)]

		for filename in filenames:
			self.print_c("Importing " + filename)

			with open(data_path + "/" + filename) as fp:

				try:
					frames = pickle.load(fp)[frame_start:frame_end]
				except ValueError:
					self.print_c("ValueError on " + filename + ", failed to load")
					continue
				except:
					self.print_c("Other error on " + filename + ", failed to load")
					continue

				new_motion = Motion(filename.replace(".pickle", ""))
				try:

					for frame in frames:
						for rb in frame.RigidBodies:
							if rb.id == CAMERA_ID:
								quat = (rb.qw, rb.qx, rb.qy, rb.qz)
								cam_dir = qv_mult(quat, (1, 0, 0))
								cam_pos = np.array([rb.x, rb.y, rb.z])
							if rb.id == OBJECT_ID:
								nofilt_pos = np.array([rb.x, rb.y, rb.z])

						latency_dist = np.dot(nofilt_pos - cam_pos, cam_dir)

						new_motion.trajectory.append(latency_dist)
						new_motion.original_trajectory.append(latency_dist)

				except ValueError:
					self.print_c("Error in frame format... Probably wrong rigidbody count, or even no rigidbody !?")
					continue

				self.motions.append(new_motion)

class MotionEditionWidget(QtGui.QWidget):
	def __init__(self):

		super(MotionEditionWidget, self).__init__()
		self.col = -1
		self.synth_count = 0
		self.print_c = None

		self.motions_db = MotionDatabase()

		self.show_kfs = True

		self.derivative_mode = 0

		self.traj_list = QtGui.QListWidget()
		self.traj_list.currentItemChanged.connect(self.list_change_event)

		self.kf_list = QtGui.QListWidget()
		self.plot = pg.PlotWidget()

		self.line = pg.LinearRegionItem(movable=True)
		self.plot.addItem(self.line)

		self.layout = QtGui.QGridLayout()

		self.delete_seq_button = QtGui.QPushButton('Delete Sequence')
		self.delete_seq_button.clicked.connect(self.delete_seq_event)

		self.add_kf_button = QtGui.QPushButton('Add KeyFrame')
		self.add_kf_button.clicked.connect(self.add_kf_event)

		self.remove_kf_button = QtGui.QPushButton('Remove KeyFrame')
		self.remove_kf_button.clicked.connect(self.remove_kf_event)

		self.remove_traj_button = QtGui.QPushButton('Remove Trajectory')
		self.remove_traj_button.clicked.connect(self.remove_traj_event)

		self.hide_kf_button = QtGui.QPushButton('Hide Keyframes')
		self.hide_kf_button.clicked.connect(self.hide_kf_clicked)

		self.derivative_button = QtGui.QPushButton('To Velocities')
		self.derivative_button.clicked.connect(self.derivative_clicked)

		self.synth_seq_button = QtGui.QPushButton('Generate Sequence')
		self.synth_seq_button.clicked.connect(self.generate_synthetic_ex)

		self.layout.addWidget(QtGui.QLabel("Motions: "), 0, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.traj_list, 1, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(QtGui.QLabel("KeyFrames: "), 2, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.kf_list, 3, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.plot, 0, 1, -1, 1, 0)

		self.layout.addWidget(self.delete_seq_button, 4, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.add_kf_button, 5, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.remove_kf_button, 6, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.remove_traj_button, 7, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.hide_kf_button, 8, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.derivative_button, 9, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.synth_seq_button, 10, 0, QtCore.Qt.AlignTop)
		verticalSpacer = QtGui.QSpacerItem(100, 1000, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
		self.layout.addItem(verticalSpacer, 11, 0, QtCore.Qt.AlignTop)

		self.setLayout(self.layout)

		self.refresh_all()

	def derivative_clicked(self):
		if self.derivative_mode == 0:
			self.derivative_mode = 1
			self.derivative_button.setText("To Acceleration")
		elif self.derivative_mode == 1:
			self.derivative_mode = 2
			self.derivative_button.setText("To Positions")
		else:
			self.derivative_mode = 0
			self.derivative_button.setText("To Velocities")

		for motion in self.motions_db.motions:

			T = motion.edited_traj

			if self.derivative_mode == 0:
				motion.trajectory = motion.edited_traj
				continue

			motion.trajectory = motion.differentiate(self.derivative_mode, True)

		self.refresh_plot()

	def hide_kf_clicked(self):
		if self.show_kfs:
			self.show_kfs = False
			self.hide_kf_button.setText("Show Keyframes")
			self.refresh_plot()
		else:
			self.show_kfs = True
			self.hide_kf_button.setText("Hide Keyframes")
			self.refresh_plot()

	def list_change_event(self, current, previous):
		self.refresh_plot()
		self.refresh_KF_List()

	def delete_seq_event(self):
		cur_traj = self.traj_list.currentRow()

		if cur_traj == -1:
			self.print_c("Delete Seq Fail: No trajectory to edit...")
			return

		[start, stop] = [int(a) for a in self.line.getRegion()]
		start = max(start,0)
		traj = self.motions_db.motions[cur_traj].edited_traj

		self.motions_db.motions[cur_traj].edited_traj = traj[0:start] + traj[stop:]

		for kf in self.motions_db.motions[cur_traj].key_frames:
			if kf > start and kf < stop:
					self.motions_db.motions[cur_traj].key_frames.remove(kf)

		for i in range(0,len(self.motions_db.motions[cur_traj].key_frames)):
			kf = self.motions_db.motions[cur_traj].key_frames[i]
			if kf > stop:
				self.motions_db.motions[cur_traj].key_frames[i] -= (stop-start)

		self.motions_db.motions[cur_traj].trajectory = self.motions_db.motions[cur_traj].edited_traj

		self.refresh_KF_List()
		self.refresh_plot()

	def add_kf_event(self):
		index = self.traj_list.currentRow()
		if index != -1:
			[start, stop] = [int(a) for a in self.line.getRegion()]
			kf_time = int((stop + start) / 2.0)
			traj_length = len(self.motions_db.motions[index].trajectory)

			kf_time = max(0,min(traj_length, kf_time))

			self.plot.addLine(kf_time)
			self.motions_db.motions[index].key_frames.append(kf_time)
			self.motions_db.motions[index].key_frames = sorted(self.motions_db.motions[index].key_frames)
			self.refresh_KF_List()
			self.refresh_plot()
			self.print_c("Added Keyframe at t=" + str(kf_time))
		else:
			self.print_c("Add Keyframe Fail: No trajectory to edit...")

	def remove_kf_event(self):
		traj_index = self.traj_list.currentRow()
		kf_index = self.kf_list.currentRow()
		if kf_index != -1 and traj_index != -1:
			kf_time = self.motions_db.motions[traj_index].key_frames[kf_index]
			self.motions_db.motions[traj_index].key_frames.remove(kf_time)
			self.refresh_KF_List()
			self.refresh_plot()
			self.print_c("Removed Keyframe at t=" + str(kf_time))

	def remove_traj_event(self):
		traj_index = self.traj_list.currentRow()
		if traj_index != -1:
			motion = self.motions_db.motions[traj_index]
			self.motions_db.motions.remove(motion)
			self.refresh_traj_list()
			self.refresh_plot()
			self.print_c("Removed traj " + motion.name)

	def refresh_plot(self):
		self.plot.clear()

		index = self.traj_list.currentRow()
		if index > -1:
			traj = self.motions_db.motions[index].trajectory
			self.plot.plot(traj)

			if self.show_kfs:
				for kf_time in self.motions_db.motions[index].key_frames:
					self.plot.addLine(kf_time)

		self.plot.addItem(self.line)

	def refresh_traj_list(self):
		i = 0
		self.traj_list.clear()
		for t in self.motions_db.motions:
			i += 1
			li = QtGui.QListWidgetItem()
			li.setText(t.name)
			self.traj_list.insertItem(i, li)

		self.traj_list.setMaximumWidth(200)

	def refresh_KF_List(self):
		self.kf_list.clear()
		index = self.traj_list.currentRow()
		if index != -1:
			i = 0
			for kf_time in self.motions_db.motions[index].key_frames:
				i += 1
				li = QtGui.QListWidgetItem()
				li.setText("KF #" + str(i) + ": " + str(kf_time))
				self.kf_list.insertItem(i, li)

		self.kf_list.setMaximumWidth(200)

	def generate_synthetic_ex(self):
		type = 2
		self.synth_count += 1
		if type == 1:
			start_mean, end_mean, start_time, end_time, s_var, e_var, s_t_var, e_t_var = 0, 10, 30, 150, 1, 1, 10, 10

			trajectory = []

			start_p = np.random.normal(start_mean, s_var)
			end_p = np.random.normal(end_mean, e_var)

			start_time = np.random.normal(start_time, s_t_var)
			end_time = np.random.normal(end_time, e_t_var)

			for i in range(0, int(end_time + 10*e_t_var)):
				if(i < start_time):
					trajectory.append(start_p)
				elif(i > start_time and i < end_time):
					trajectory.append(start_p + ((end_p - start_p)/(end_time-start_time)) * (i - start_time))
				else:
					trajectory.append(end_p)
			traj_name = "Synthesised[" + str(int(start_p)) + "-" + str(int(end_p)) + "]"

		elif type == 2:
			start_mean, end_mean, start_time, end_time, s_var, e_var, s_t_var, e_t_var = 0, 100, 30, 150, 10, 10, 10, 10

			gaussian_event_time_mean, gaussian_event_time_var = 70, 10
			gaussian_event_width_mean, gaussian_event_width_var = 10, 1
			gaussian_event_height_mean, gaussian_event_height_var = 200, 1

			trajectory = []

			start_p = np.random.normal(start_mean, s_var)
			end_p = np.random.normal(end_mean, e_var)

			start_time = np.random.normal(start_time, s_t_var)
			end_time = np.random.normal(end_time, e_t_var)

			gaussian_event_time = np.random.normal(gaussian_event_time_mean, gaussian_event_time_var)
			gaussian_event_width = np.random.normal(gaussian_event_width_mean, gaussian_event_width_var)
			gaussian_event_height = np.random.normal(gaussian_event_height_mean, gaussian_event_height_var)

			for i in range(0, int(end_time + 10 * e_t_var)):
				if (i < start_time):
					trajectory.append(start_p)
				elif (i > start_time and i < end_time):
					trajectory.append(start_p + ((end_p - start_p) / (end_time - start_time)) * (i - start_time))
				else:
					trajectory.append(end_p)

				trajectory[-1] += gaussian_event_height * stats.norm.pdf(i, gaussian_event_time, gaussian_event_width)

			traj_name = "Synthesised_" + str(self.synth_count) + "_[" + str(int(start_p)) + "-" + str(int(end_p)) + "]"


		self.motions_db.add_motion(traj_name, trajectory)
		self.refresh_all()

	def refresh_all(self):
		self.refresh_traj_list()
		self.refresh_KF_List()
		self.refresh_plot()

class LearningTrajectoryWidget(QtGui.QWidget):
	def __init__(self):

		self.col = -1

		self.registered_ref = -1

		super(LearningTrajectoryWidget, self).__init__()

		self.print_c = None

		self.motions_db = MotionDatabase()

		self.plot = pg.PlotWidget()

		self.layout = QtGui.QGridLayout()

		self.compute_reg_button = QtGui.QPushButton('Learn Reference Trajectory')
		self.compute_reg_button.clicked.connect(self.compute_reference)

		self.layout.addWidget(self.plot, 0, 1, -1, 1, 0)

		verticalSpacer = QtGui.QSpacerItem(100, 3000, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
		self.layout.addItem(verticalSpacer, 5, 0, QtCore.Qt.AlignTop)

		self.setLayout(self.layout)

		self.refresh_all()

	class ReferenceTrajectory(object):
		def __init__(self):
			self.traj_length = 0
			self.series = np.zeros(2,1)

	def compute_reference(self):

		# for reg_traj in self.motions_db.motions:
		# 	if reg_traj == ref_motion:
		# 		continue
		#
		# 	warp = DynamicTimeWarp()
		#
		# 	if(not warp.compute_warp(np.array(ref_motion.edited_traj).reshape(-1,1), np.array(reg_traj.edited_traj).reshape(-1,1))):
		# 		errors += 1
		#
		# 	reg_traj.time_warp = warp
		#
		# 	self.print_c("Completed Registration of " + reg_traj.name)
		#
		# self.print_c("Registrations Computed for " + str(len(self.motions_db.motions)) + " motions, " + str(errors) + " errors.")

		self.refresh_plot()

	def refresh_plot(self):
		self.plot.clear()

	def refresh_all(self):
		self.refresh_plot()

class OnlineRegistrationWidget(QtGui.QWidget):
	def __init__(self):

		self.col = -1

		self.is_show_splines = False
		self.is_registering = False
		self.is_prediction_showcase = False

		self.registered_ref = -1

		super(OnlineRegistrationWidget, self).__init__()

		self.print_c = None

		self.motions_db = MotionDatabase()

		self.traj_list_ref = QtGui.QListWidget()
		self.traj_list_ref.currentItemChanged.connect(self.list_change_event)

		self.traj_list_reg = QtGui.QListWidget()
		self.traj_list_reg.currentItemChanged.connect(self.list_change_event)

		self.ref_plot = pg.PlotWidget()
		self.input_plot = pg.PlotWidget()

		self.layout = QtGui.QGridLayout()

		self.compute_reg_button = QtGui.QPushButton('Online Registration showcase')
		self.compute_reg_button.pressed.connect(self.one_on_one_showcase)

		self.vel_pred_button = QtGui.QPushButton('Velocity Prediction showcase')
		self.vel_pred_button.pressed.connect(self.velocity_prediction_showcase)

		self.pos_pred_button = QtGui.QPushButton('Position Prediction showcase')
		self.pos_pred_button.pressed.connect(self.position_prediction_showcase)

		self.kal_pos_pred_button = QtGui.QPushButton('Kalman Prediction showcase')
		self.kal_pos_pred_button.pressed.connect(self.kalman_position_prediction_showcase)

		self.show_warp_btn = QtGui.QPushButton('Show Warp')
		self.show_warp_btn.clicked.connect(self.show_splines)

		self.layout.addWidget(QtGui.QLabel("Reference Motions: "), 0, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.traj_list_ref, 1, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.compute_reg_button, 2, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.vel_pred_button, 3, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.pos_pred_button, 4, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.kal_pos_pred_button, 5, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.show_warp_btn, 6, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(QtGui.QLabel("Register Motion: "), 7, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.traj_list_reg, 8, 0, QtCore.Qt.AlignTop)
		self.layout.addWidget(self.ref_plot, 0, 1, -1, 1, 0)
		self.layout.addWidget(self.input_plot, 0, 2, -1, 1, 0)

		verticalSpacer = QtGui.QSpacerItem(100, 3000, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
		self.layout.addItem(verticalSpacer, 9, 0, QtCore.Qt.AlignTop)

		self.setLayout(self.layout)

		self.current_warp = None

		self.refresh_all()

	def one_on_one_showcase(self):
		errors = 0
		ref_indx = self.traj_list_ref.currentRow()
		self.registered_ref = ref_indx
		if ref_indx == -1:
			self.print_c("No Reference Trajectory Selected")
			return

		ref_motion = self.motions_db.motions[ref_indx]

		reg_indx = self.traj_list_reg.currentRow()

		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		input_motion = self.motions_db.motions[reg_indx]

		if self.current_warp == None or self.current_warp.input_motion != input_motion or self.current_warp.reference_motion != ref_motion:
			self.current_warp = GreedyWarp(ref_motion, input_motion, 50)

		self.is_registering = True
		while len(self.current_warp.evaluated_warp) < len(input_motion.trajectory):
			self.current_warp.register_next_frame()
			self.refresh_plot()
			self.print_c("")
			time.sleep(0.05)
		self.is_registering = False

	def velocity_prediction_showcase(self):
		reg_indx = self.traj_list_reg.currentRow()

		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		input_motion = self.motions_db.motions[reg_indx]

		ref_motions_warps = {}

		for m in self.motions_db.motions:
			if m.name != input_motion.name:
				ref_motions_warps[m.name] = GreedyWarp(m, input_motion, 50)

		timings = []
		self.is_prediction_showcase = True
		for i in range(0,len(input_motion.trajectory) - 1):
			time.sleep(0.00)
			now = time.time()
			vel_estimates = []
			for warp in ref_motions_warps.values():
				warp.register_next_frame()

				if warp.evaluated_warp[-1] + 1 < len(warp.ref_feature_list):
					vel_estimate = warp.ref_feature_list[warp.evaluated_warp[-1] + 1][1]
				else:
					vel_estimate = warp.ref_feature_list[warp.evaluated_warp[-1]][1]

				vel_estimates.append([1,vel_estimate])

			gmm = GaussianMixture(n_components=1)
			gmm.fit(vel_estimates)

			#avg_vel_estimate = np.average(vel_estimates)
			avg_vel_estimate = gmm.means_[0][1]

			timings.append(time.time() - now)

			self.input_plot.clear()
			self.ref_plot.clear()

			self.input_plot.plot(input_motion.trajectory)
			self.input_plot.addLine(i)
			self.ref_plot.plot([0,avg_vel_estimate])

			self.print_c("")
		self.print_c("Average time for velocity prediction: " + str(np.average(timings)))
		self.is_registering = False

	def position_prediction_showcase(self):
		reg_indx = self.traj_list_reg.currentRow()

		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		input_motion = self.motions_db.motions[reg_indx]

		ref_motions_warps = {}

		for m in self.motions_db.motions:
			if m.name != input_motion.name:
				ref_motions_warps[m.name] = GreedyWarp(m, input_motion, 50)

		trajectory = [input_motion.trajectory[0],input_motion.trajectory[0],input_motion.trajectory[0],input_motion.trajectory[0]]

		timings = []
		self.is_prediction_showcase = True
		agg_vel_ests = []
		averaged_estimate = 0
		mv_avg_win_size = 2
		for i in range(4, len(input_motion.trajectory) - 1):
			time.sleep(0.00)
			now = time.time()
			vel_estimates = []
			track_speed = []
			pos_estimates = []
			confidences = []
			for warp in ref_motions_warps.values():
				old_warped_pos = warp.evaluated_warp[-1]
				warp.register_next_frame()

				if warp.evaluated_warp[-1] + 4 < len(warp.ref_feature_list):
					vel_estimate = warp.reference_motion.trajectory[warp.evaluated_warp[-1] + 4] - warp.reference_motion.trajectory[warp.evaluated_warp[-1]]
					track_speed = warp.evaluated_warp[-1] - old_warped_pos
					pos_estimate = warp.reference_motion.trajectory[warp.evaluated_warp[-1] + 4]
				else:
					vel_estimate = 0
					pos_estimate = warp.reference_motion.trajectory[warp.evaluated_warp[-1]]
				vel_estimates.append([1,track_speed * vel_estimate])
				pos_estimates.append(pos_estimate)
				confidences.append(1 / np.sqrt(warp.match_error))

			gmm = GaussianMixture(n_components=1)
			gmm.fit(vel_estimates)

			timings.append(time.time() - now)

			pos_estimates = np.array(pos_estimates)
			confidences = np.array(confidences)

			vels = [v[1] for v in vel_estimates]
			agg_vel_est = np.dot(vels, confidences) / sum(confidences)

			agg_vel_ests.append(agg_vel_est)

			if len(trajectory) - 4 < mv_avg_win_size:
				averaged_estimate = agg_vel_est
			else:
				averaged_estimate = averaged_estimate + agg_vel_ests[-1]/ mv_avg_win_size - agg_vel_ests[-(mv_avg_win_size + 1)] / mv_avg_win_size

			if abs(input_motion.trajectory[i] - input_motion.trajectory[i-1]) < 0.001:
				averaged_estimate = 0

			trajectory.append(input_motion.trajectory[i-4] + averaged_estimate)
			self.input_plot.clear()
			self.ref_plot.clear()

			self.input_plot.plot(input_motion.trajectory)
			self.input_plot.addLine(i)
			self.ref_plot.plot(trajectory)

			self.print_c("")

		blur_plot = pg.PlotWidget()
		blur_plot.plot(get_blur_trajectory(trajectory, input_motion.trajectory))
		blur_plot.setWindowTitle("Blur Trajectory")
		blur_plot.show()

		self.print_c("Average time for velocity prediction: " + str(np.average(timings)))
		self.is_registering = False

	def kalman_position_prediction_showcase(self):
		reg_indx = self.traj_list_reg.currentRow()

		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		input_motion = self.motions_db.motions[reg_indx]

		ref_motions_warps = {}

		for m in self.motions_db.motions:
			if m.name != input_motion.name:
				ref_motions_warps[m.name] = GreedyWarp(m, input_motion, 50)

		trajectory = []
		timings = []
		self.is_prediction_showcase = True

		LATENCY, DT, OBS_VAR, PROC_VAR = 0.04, 0.01, 0.000001, 100

		p_e, v_e, a_e = np.array([0.5 * DT ** 2, DT, 2.0]) * PROC_VAR

		initial_state = np.array([[input_motion.trajectory[0], 0, 0], [0, 0, 0], [0, 0, 0]])
		initial_cov = np.diag([p_e, v_e, a_e])
		dt_matrix = np.roll(np.diag([DT, DT, 0]), 1)
		transition_matrix = np.eye(3) + dt_matrix
		observation_cov = np.diag([OBS_VAR])
		observation_matrix = np.array([1, DT, DT * DT])

		state_mean = [initial_state]
		state_cov = [initial_cov]

		kf = KalmanFilter (
			transition_matrices=transition_matrix,
			observation_matrices=observation_matrix,
			initial_state_mean=initial_state,
			initial_state_covariance=initial_cov,
			transition_covariance=initial_cov,
			observation_covariance=observation_cov
		)
		agg_pos_ests = []
		mv_avg_win_size = 4
		for i in range(4, len(input_motion.trajectory) - 1):
			time.sleep(0.00)
			now = time.time()

			if len(state_mean) == 1:
				state_mean[0][0] = np.array(input_motion.trajectory[i-4])

			new_state, new_cov = kf.filter_update(
				np.array(state_mean[-1]),
				np.array(state_cov[-1]),
				np.array([np.array(input_motion.trajectory[i-4])])
			)

			state_mean.append(new_state)
			state_cov.append(new_cov)

			filtered_obj_pos = new_state[0] + new_state[1] * LATENCY

			timings.append(time.time() - now)

			trajectory.append(filtered_obj_pos[0])
			self.input_plot.clear()
			self.ref_plot.clear()

			self.input_plot.plot(input_motion.trajectory)
			self.input_plot.addLine(i)
			self.ref_plot.plot(trajectory)

			self.print_c("")

		blur_plot = pg.PlotWidget()
		blur_plot.plot(get_blur_trajectory(trajectory, input_motion.trajectory[4:]))
		blur_plot.setWindowTitle("Blur Trajectory")
		blur_plot.show()

		self.print_c("Average time for velocity prediction: " + str(np.average(timings)))
		self.is_registering = False

	def show_warp(self):
		self.is_show_splines = False
		self.show_warp_btn.setText('Show Warp Splines')
		self.show_warp_btn.clicked.connect(self.show_splines)
		self.refresh_plot()

	def show_splines(self):
		self.is_show_splines = True
		self.show_warp_btn.setText('Show Warp Trajs')
		self.show_warp_btn.clicked.connect(self.show_warp)
		self.refresh_plot()

	def list_change_event(self, current, previous):
		self.refresh_plot()

	def refresh_plot(self):
		self.ref_plot.clear()
		self.input_plot.clear()

		if self.is_registering and self.current_warp != None:
			self.ref_plot.plot(self.current_warp.normalize_list(self.current_warp.reference_motion.trajectory))
			self.input_plot.plot(self.current_warp.normalize_list(self.current_warp.input_motion.trajectory))

			cur_frame = len(self.current_warp.evaluated_warp) - 1

			if cur_frame > -1:
				self.input_plot.addLine(cur_frame)
				self.ref_plot.addLine(self.current_warp.evaluated_warp[cur_frame])

		elif self.current_warp != None:
			if self.is_show_splines:
				self.ref_plot.clear()
				self.input_plot.plot(self.current_warp.evaluated_warp)
			else:
				self.input_plot.plot(self.current_warp.input_motion.trajectory)
				self.ref_plot.plot(self.current_warp.warp_motion(self.current_warp.reference_motion.trajectory))

	def refresh_traj_list(self):
		i = 0
		self.traj_list_reg.clear()
		self.traj_list_ref.clear()
		for t in self.motions_db.motions:
			i += 1
			li1 = QtGui.QListWidgetItem()
			li1.setText(t.name)
			li2 = QtGui.QListWidgetItem()
			li2.setText(t.name)
			self.traj_list_ref.insertItem(i, li1)
			self.traj_list_reg.insertItem(i, li2)

		self.traj_list_ref.setMaximumWidth(200)
		self.traj_list_ref.setMaximumHeight(300)
		self.traj_list_reg.setMaximumWidth(200)
		self.traj_list_reg.setMaximumHeight(300)

	def refresh_all(self):
		self.refresh_traj_list()
		self.refresh_plot()

class GuiApp(object):

	def __init__(self):

		self.columns = [False, False, False, False]

		self.print_c = self.print_to_console

		self.motions_db = MotionDatabase()

		self.motions_db.print_c = self.print_to_console

		self.motion_edit_widget = None
		self.natnet_widget = None
		self.time_registration_widget = None
		self.compute_eigen_widget = None
		self.filter_traj_widget = None
		self.online_filter_traj_widget = None
		self.online_registration_widget = None

		## Always start by initializing Qt (only once per application)
		self.app = QtGui.QApplication([])
		## Define a top-level widget to hold everything
		self.window = QtGui.QMainWindow()
		self.app.setActiveWindow(self.window)
		self.window.resize(1280,720)

		self.window.setWindowTitle("Deformable Motions Editor")

		self.widget = QtGui.QWidget()

		# Create main menu
		self.setup_menu()

		self.consoleText = QtGui.QTextEdit()
		self.consoleText.setText("Console Output Below:\n")
		self.consoleText.setReadOnly(True)
		self.consoleText.setMaximumHeight(200)

		self.layout = QtGui.QGridLayout()
		self.widget.setLayout(self.layout)

		self.layout.addWidget(self.consoleText, 1, 0, 1, -1,QtCore.Qt.AlignBottom)

		self.window.setCentralWidget(self.widget)
		self.window.show()

		timer = QtCore.QTimer(self.app)
		timer.timeout.connect(self.update_all)
		timer.start(10)

	def setup_menu(self):
		mainMenu = self.window.menuBar()
		mainMenu.setNativeMenuBar(False)
		fileMenu = mainMenu.addMenu('File')
		viewMenu = mainMenu.addMenu('View')
		editMenu = mainMenu.addMenu('Edit')

		# Add load folder button
		ldFolderButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Load Mocap', self.window)
		ldFolderButton.setShortcut('CTRL+O')
		ldFolderButton.setStatusTip('Load Mocap data')
		ldFolderButton.triggered.connect(self.load_motions)
		fileMenu.addAction(ldFolderButton)

		# Add Import DB button
		impDBButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Import Motions', self.window)
		impDBButton.setShortcut('CTRL+SHIFT+O')
		impDBButton.setStatusTip('Load Motions Database')
		impDBButton.triggered.connect(self.import_motion_db)
		fileMenu.addAction(impDBButton)

		# Add Save DB button
		saveDBButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Save Motions', self.window)
		saveDBButton.setShortcut('CTRL+S')
		saveDBButton.setStatusTip('Save Motions Database')
		saveDBButton.triggered.connect(self.save_motions_db)
		fileMenu.addAction(saveDBButton)

		saveDBAsButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Save Motions As', self.window)
		saveDBAsButton.setShortcut('CTRL+SHIFT+S')
		saveDBAsButton.setStatusTip('Save Motions Database')
		saveDBAsButton.triggered.connect(self.save_motions_db_as)
		fileMenu.addAction(saveDBAsButton)

		# Add exit button
		exitButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Exit', self.window)
		exitButton.setShortcut('ALT+F4')
		exitButton.setStatusTip('Exit application')
		exitButton.triggered.connect(self.window.close)
		fileMenu.addAction(exitButton)

		# View Menu !
		moEditButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Motion Editor', self.window)
		moEditButton.setShortcut('CTRL+E')
		moEditButton.setStatusTip('toggle Motion Editor')
		moEditButton.triggered.connect(self.toggle_motion_edit_widget)
		viewMenu.addAction(moEditButton)

		olRegButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Online Registration', self.window)
		olRegButton.setShortcut('CTRL+T')
		olRegButton.setStatusTip('toggle Online Registration')
		olRegButton.triggered.connect(self.toggle_online_registration_widget)
		viewMenu.addAction(olRegButton)

		timeRegButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Time Registration', self.window)
		timeRegButton.setShortcut('CTRL+Y')
		timeRegButton.setStatusTip('toggle Time Registration')
		timeRegButton.triggered.connect(self.toggle_time_registration_widget)
		viewMenu.addAction(timeRegButton)

		nnAcquButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Natnet', self.window)
		nnAcquButton.setShortcut('CTRL+N')
		nnAcquButton.setStatusTip('Toggle Natnet Acquisition')
		nnAcquButton.triggered.connect(self.toggle_natnet_widget)
		viewMenu.addAction(nnAcquButton)

		self.addKFButMenu = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Add KeyFrame', self.window)
		self.addKFButMenu.setShortcut('K')
		self.addKFButMenu.setStatusTip('Add KeyFrame')
		self.addKFButMenu.setDisabled(True)
		editMenu.addAction(self.addKFButMenu)

		self.delButMenu = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Delete Sequence', self.window)
		self.delButMenu.setShortcut('Del')
		self.delButMenu.setStatusTip('Delete Sequence')
		self.delButMenu.setDisabled(True)
		editMenu.addAction(self.delButMenu)

	def toggle_motion_edit_widget(self):
		if self.motion_edit_widget == None:
			self.motion_edit_widget = MotionEditionWidget()
			self.motion_edit_widget.motions_db = self.motions_db
			self.motion_edit_widget.print_c = self.print_c
			self.motion_edit_widget.refresh_all()
			col = self.assign_col()
			self.motion_edit_widget.col = col
			self.layout.addWidget(self.motion_edit_widget,0, col, 1, 1, 0)
			self.addKFButMenu.setDisabled(False)
			self.addKFButMenu.triggered.connect(self.motion_edit_widget.add_kf_event)
			self.delButMenu.setDisabled(False)
			self.delButMenu.triggered.connect(self.motion_edit_widget.delete_seq_event)

		else:
			self.columns[self.motion_edit_widget.col] = False
			self.layout.removeWidget(self.motion_edit_widget)
			self.motion_edit_widget.deleteLater()
			self.motion_edit_widget = None
			self.addKFButMenu.setDisabled(True)
			self.delButMenu.setDisabled(True)

	def toggle_time_registration_widget(self):
		if self.time_registration_widget == None:
			self.time_registration_widget = LearningTrajectoryWidget()
			self.time_registration_widget.motions_db = self.motions_db
			self.time_registration_widget.print_c = self.print_c
			self.time_registration_widget.refresh_all()
			col = self.assign_col()
			self.time_registration_widget.col = col
			self.layout.addWidget(self.time_registration_widget,0, col, 1, 1, 0)
		else:
			self.columns[self.time_registration_widget.col] = False
			self.layout.removeWidget(self.time_registration_widget)
			self.time_registration_widget.deleteLater()
			self.time_registration_widget = None

	def toggle_online_registration_widget(self):
		if self.online_registration_widget == None:
			self.online_registration_widget = OnlineRegistrationWidget()
			self.online_registration_widget.motions_db = self.motions_db
			self.online_registration_widget.print_c = self.print_c
			self.online_registration_widget.refresh_all()
			col = self.assign_col()
			self.online_registration_widget.col = col
			self.layout.addWidget(self.online_registration_widget, 0, col, 1, 1, 0)
		else:
			self.columns[self.online_registration_widget.col] = False
			self.layout.removeWidget(self.online_registration_widget)
			self.online_registration_widget.deleteLater()
			self.online_registration_widget = None

	# def toggle_online_filter_traj_widget(self):
	# 	if self.online_filter_traj_widget == None:
	# 		self.online_filter_traj_widget = OnlineFilteringTrajectoryWidget()
	# 		self.online_filter_traj_widget.motions_db = self.motions_db
	# 		self.online_filter_traj_widget.print_c = self.print_c
	# 		self.online_filter_traj_widget.refresh_all()
	# 		col = self.assign_col()
	# 		self.online_filter_traj_widget.col = col
	# 		self.layout.addWidget(self.online_filter_traj_widget,0, col, 1, 1, 0)
	# 	else:
	# 		self.columns[self.online_filter_traj_widget.col] = False
	# 		self.layout.removeWidget(self.online_filter_traj_widget)
	# 		self.online_filter_traj_widget.deleteLater()
	# 		self.online_filter_traj_widget = None

	def toggle_natnet_widget(self):
		if self.natnet_widget == None:
			self.natnet_widget = NatNetAcquisitionWidget()
			self.natnet_widget.print_c = self.print_to_console
			col = self.assign_col()
			self.natnet_widget.col = col
			self.layout.addWidget(self.natnet_widget, 0, col, 1, 1)
		else:

			self.columns[self.natnet_widget.col] = False
			self.natnet_widget.done()

	def update_all(self):
		if self.natnet_widget != None:
			self.natnet_widget.update_plot()
			if self.natnet_widget.shouldImport:
				date = time.strftime('[%d-%m-%y] [%Hh-%Mm-%Ss]')
				name = "{}".format(date)
				self.print_c("Importing \"" + name + "\" from NatNet")
				self.motions_db.add_motion(name, self.natnet_widget.acquisition.plot)
				self.natnet_widget.acquisition.plot = []
				self.natnet_widget.plot.clear()
				self.natnet_widget.shouldImport = False

				if self.motion_edit_widget != None:
					self.motion_edit_widget.refresh_all()
			if self.natnet_widget.shouldExit:
				self.columns[self.natnet_widget.col] = False
				self.layout.removeWidget(self.natnet_widget)
				self.natnet_widget.deleteLater()
				#self.natnet_widget.__del__()
				self.natnet_widget = None

		if self.compute_eigen_widget != None:
			self.compute_eigen_widget.refresh_button()

	def load_motions(self):
		file_selection = QtGui.QFileDialog()
		file_selection.setAcceptMode(file_selection.AcceptOpen)
		file_selection.setFileMode(file_selection.ExistingFiles)
		file_selection.setNameFilters(["Mocap Data (*" + ".pickle" + ")"])
		file_selection.show()
		data_paths = []
		if file_selection.exec_():
			fileNames = file_selection.selectedFiles()

			for f in fileNames:
				path = "/".join(f.split("/")[:-1])
				self.motions_db.load(path,[f.split("/")[-1]])

			if self.motion_edit_widget != None:
				self.motion_edit_widget.refresh_all()
			if self.time_registration_widget != None:
				self.time_registration_widget.refresh_all()

	def save_motions_db_as(self):
		file_selection = QtGui.QFileDialog()
		file_selection.setAcceptMode(file_selection.AcceptSave)
		file_selection.setFileMode(file_selection.AnyFile)
		file_selection.setNameFilters(["Motion Database (*" + DB_FILE_EXTENSION + ")"])
		file_selection.show()
		self.motions_db.dbFilename = ""
		if file_selection.exec_():
			self.motions_db.dbFilename = file_selection.selectedFiles()[0]
			if self.motions_db.dbFilename[-len(DB_FILE_EXTENSION):] != DB_FILE_EXTENSION:
				self.motions_db.dbFilename += DB_FILE_EXTENSION

		if self.motions_db.dbFilename == "":
			return
		self.print_c("Saving DB: " + self.motions_db.dbFilename)
		call_back_temp = self.motions_db.print_c
		self.motions_db.print_c = None

		with open(self.motions_db.dbFilename, 'w') as fp:
			pickle.dump(self.motions_db, fp)

		self.motions_db.print_c = call_back_temp

	def save_motions_db(self):
		if self.motions_db.dbFilename == "":
			self.save_motions_db_as()
		else:
			self.print_c("Saving DB: " + self.motions_db.dbFilename)
			call_back_temp = self.motions_db.print_c
			motion_model_temp = self.motions_db.motion_model
			self.motions_db.print_c = None
			self.motions_db.motion_model = None

			timewarps = [m.time_warp for m in self.motions_db.motions]
			for m in self.motions_db.motions:
				m.time_warp = None

			with open(self.motions_db.dbFilename, 'w') as fp:
				pickle.dump(self.motions_db, fp)

			for m in range(0,len(self.motions_db.motions)):
				self.motions_db.motions[m].time_warp = timewarps[m]

			self.motions_db.print_c = call_back_temp
			self.motions_db.motion_model = motion_model_temp

	def import_motion_db(self):
		filename = ""
		file_selection = QtGui.QFileDialog()
		file_selection.setAcceptMode(file_selection.AcceptOpen)
		file_selection.setFileMode(file_selection.ExistingFile)
		file_selection.setNameFilters(["Motion Database (*" + DB_FILE_EXTENSION + ")"])
		file_selection.show()

		if file_selection.exec_():
			filename = file_selection.selectedFiles()[0]

		if filename == "":
			return

		self.print_c("Loading DB: " + filename)
		with open(filename, 'r') as fp:
			self.motions_db = pickle.load(fp)

		self.motions_db.print_c = self.print_c

		if self.motion_edit_widget != None:
			self.motion_edit_widget.motions_db = self.motions_db
			self.motion_edit_widget.refresh_all()
		if self.time_registration_widget != None:
			self.time_registration_widget.motions_db = self.motions_db
			self.time_registration_widget.refresh_all()

	def print_to_console(self,string):
		self.consoleText.setText(self.consoleText.toPlainText() + string + "\n")
		scrollbar = self.consoleText.verticalScrollBar()
		scrollbar.setValue(scrollbar.maximum())
		self.app.processEvents()

	def assign_col(self):
		for i in range(0,len(self.columns)):
			if self.columns[i] == False:
				self.columns[i] = True
				return i

def main():
	gui = GuiApp()
	gui.app.exec_()

if __name__ == "__main__":
	main()
