import os
from utils import qv_mult
import pickle
import numpy as np
import c3d, PyBVH

CAMERA_ID = 1
OBJECT_ID = 2
DB_FILE_EXTENSION = ".modb"

class Motion3D(object):
	def __init__(self, name):
		self.print_c = None

		self.scale = 1.0
		self.name = name
		self.markers = []
		self.original_trajectory = []
		self.trajectory = []
		self.time_step = 1

		self.bones = []
		self.rigid_parts_names = []
		self.rigid_transforms = []
		self.n_squared_twists_trajectories = dict()

		self.original_file_format = ".none"

	def differentiate(self, order, smooth=False):
		gauss_kernel = np.loadtxt("gauss_kernel.csv", delimiter=",")
		T = self.trajectory
		derivative = np.zeros((len(T),3*len(self.markers)))
		for i in range(1, len(T) - 1):
			if order == 1:
				# derivative[i] = (-T[i + 2] + 8 * T[i + 1] - 8 * T[i - 1] + T[i - 2]) / 12.0
				derivative[i] = T[i] - T[i - 1]
			elif order == 2:
				# derivative[i] = (-T[i + 2] + 16 * T[i + 1] - 30 * T[i] + 16 * T[i - 1] - T[i - 2]) / 12.0
				derivative[i] = (T[i - 1] - 2 * T[i] + T[i + 1])
			else:
				self.print_c("Derivative of order: " + str(order) + " not supported.")

		if smooth:

			unpacked_traj = np.transpose(derivative)

			convolved_dimensions = []
			for dim_traj in unpacked_traj:
				convolve_dim_traj = np.convolve(dim_traj, gauss_kernel)
				convolved_dimensions.append(convolve_dim_traj)

			derivative = np.transpose(convolved_dimensions)[:len(self.trajectory)]

		return derivative


class Motion3DDatabase(object):
	def __init__(self):
		self.print_c = None

		self.dbFilename = ""
		self.reference_motion = "None"
		self.motions = []
		self.motion_model = None

	def add_motion(self, name, motion_traj, markers_dimension, correct_for_natnet):
		new_traj = Motion3D(name)
		new_traj.markers = ["Marker 1"]
		if correct_for_natnet:
			new_traj.trajectory = np.array([[m[0],m[2],m[1]] for m in motion_traj])
		else:
			new_traj.trajectory = motion_traj

		new_traj.original_trajectory = new_traj.trajectory

		self.motions.append(new_traj)

	def load_c3dtxt(self, filepath, filename):
		file = filepath + "/" + filename

		## Open the file with read only permit
		f = open(file, "r")

		## use readlines to read all lines in the file
		## The variable "lines" is a list containing all lines
		lines = f.readlines()

		markers_str = lines[1].replace("Frames\t", "").split("\t\t")

		markers = []
		for m_str in markers_str:
			m = m_str.split("\t")[0].replace(" X","")
			markers.append(m)

		marker_count = len(markers)

		data_scale = 0.001

		trajectory = []
		for ln in range(2, len(lines)):
			line = lines[ln].replace("   " + str(ln - 1) + "\t     ", "")
			m_positions = line.split("\t\t")
			flattened = []
			for vec3 in m_positions:
				vec = [float(a)*data_scale for a in vec3.split("\t")[:3]]
				corrected_vec = [vec[2],vec[0],vec[1]]
				flattened.extend(corrected_vec)

			trajectory.append(flattened)

		f.close()

		new_motion = Motion3D(filename.replace(".c3dtxt", ""))
		new_motion.markers = markers
		new_motion.original_trajectory = np.array(trajectory)
		new_motion.trajectory = new_motion.original_trajectory

		new_motion.original_file_format = ".c3dtxt"

		#self.print_c("Imported " + str(len(trajectory)) + " frames for " + str(marker_count) + " markers.")

		self.motions.append(new_motion)

	def load_c3d(self, filepath, filename):
		with open(filepath + '/' + filename, 'rb') as handle:
			reader = c3d.Reader(handle)
			frames = [a for a in reader.read_frames()]

			markers = reader.point_labels

			trajectory = []
			for (frame, points, analogs) in frames:
				data_scale = 0.005
				frame_point = []
				for p in points:
					frame_point.extend([data_scale* a for a in p[:3]])
				trajectory.append(frame_point)

		discard_doa_markers = True

		#Filter data
		if discard_doa_markers:
			transposed_traj = []
			killed_markers = []
			dims = np.transpose(trajectory)
			for i in range(0, len(markers)):
				if dims[3 * i + 0][0] == 0.0 or \
								dims[3 * i + 1][0] == 0.0 or \
								dims[3 * i + 2][0] == 0.0:
					killed_markers.append(markers[i])
				else:
					transposed_traj.extend(dims[3 * i:3 * (i + 1)])

			trajectory = np.transpose(transposed_traj)

			markers = [m for m in markers if m not in killed_markers]

		do_filter = True

		if do_filter:
			transposed_traj = []
			for dim in np.transpose(trajectory):
				i = -1
				while i < len(dim)-1:
					i += 1
					if dim[i] == 0.0:
						end_of_drop = min(i + 1, len(dim)-1)
						for j in range(i+1,len(dim)):
							if dim[j] != 0.0:
								end_of_drop = j
								break
							if j == len(dim)-1:
								end_of_drop = -1
								break
						if end_of_drop == -1:
							for j in range(i,len(dim)):
								dim[j] = dim[i-1]
						drop_length = float(end_of_drop - i)
						for j in range(i,end_of_drop):
							dim[j] = ((j-i)/drop_length) * dim[end_of_drop] + (1-((j-i)/drop_length)) * dim[i-1]
						i = end_of_drop
				transposed_traj.append(dim)

			trajectory = np.transpose(transposed_traj)

		new_motion = Motion3D(filename.replace(".c3d", ""))
		new_motion.markers = markers
		new_motion.original_trajectory = np.array(trajectory)
		new_motion.trajectory = new_motion.original_trajectory

		new_motion.original_file_format = ".c3d"

		#self.print_c("Imported " + str(len(trajectory)) + " frames for " + str(len(markers)) + " markers.")

		self.motions.append(new_motion)

	def load_bvh(self, file_path, filename):
		fp = file_path + '/' + filename

		#self.print_c("Loading BVH File " + fp)

		bvh_file = PyBVH.LoadBVH(fp)
		bvh_file.SetNormalizedScaleMultiplied(0.1)
		trajectory = []
		for i in range(0, bvh_file.GetFrameCount()):
			frame = []
			for joint in bvh_file.GetJointPositions(i, add_root=True):
				frame.append(joint[0])
				frame.append(joint[2])
				frame.append(joint[1])

			trajectory += [frame]

		jointNames = bvh_file.GetJointNames()

		new_motion = Motion3D(filename.replace(".bvh", ""))
		new_motion.markers = [jointNames[i] for i in range(int(len(trajectory[0])/3))]
		new_motion.original_trajectory = np.array(trajectory)
		new_motion.trajectory = new_motion.original_trajectory

		new_motion.time_step = 1.0 / bvh_file.GetSamplingRate()

		transforms = bvh_file.GetCumulativeTransforms(0, add_root=True)

		new_motion.scale = bvh_file.GetNormalizedScale()

		new_motion.bones = bvh_file.GetBonesByNames()
		new_motion.rigid_parts_names = [name for name in transforms]
		new_motion.rigid_transforms = []
		for i in range(0, bvh_file.GetFrameCount()):
			new_motion.rigid_transforms += [bvh_file.GetCumulativeTransforms(i, add_root=True)]

		self.print_c("Computing n_^2 twist trajectories (this may take a while) ... ")

		new_motion.n_squared_twists_trajectories = n_squared_twist_from_rigid_trajectories(new_motion, use_parallel=True)

		new_motion.original_file_format = ".bvh"

		self.motions.append(new_motion)

		self.print_c("Done loading " + fp)


def twist_from_rigid_trajectories(motion):
	if len(motion.rigid_parts_names) == 0 or len(motion.rigid_transforms) == 0:
		return []

	twists_dict = dict()
	for part_name in motion.rigid_parts_names:
		twists = []
		for i in range(1, len(motion.rigid_transforms)):
			cur_transform = np.transpose(motion.rigid_transforms[i][part_name])
			prev_transform = np.transpose(motion.rigid_transforms[i - 1][part_name])

			cur_origin = cur_transform[0:3, 3]
			prev_origin = prev_transform[0:3, 3]
			cur_origin = motion.scale * np.array([cur_origin[0], cur_origin[2], cur_origin[1]])
			prev_origin = motion.scale * np.array([prev_origin[0], prev_origin[2], prev_origin[1]])

			origin_velocity = (cur_origin - prev_origin) / 0.01

			rotation = cur_transform[0:3, 0:3]

			angular_velocity_matrix = _log_rotation(rotation)

			angular_velocity = np.array([angular_velocity_matrix[2, 1],
			                             angular_velocity_matrix[0, 2],
			                             angular_velocity_matrix[1, 0]])

			twist = origin_velocity.tolist()
			twist += (origin_velocity + angular_velocity_matrix.dot(cur_origin)).tolist()

			twists += [twist]

		twists_dict[part_name] = twists

	return twists_dict


def n_squared_twist_from_rigid_trajectory(job_packet):
	(motion, (first_part, second_part)) = job_packet

	twists = []
	for i in range(1, len(motion.rigid_transforms)):
		cur_transform = _inverse_transform(np.transpose(
			motion.rigid_transforms[i][first_part])).dot(
			np.transpose(motion.rigid_transforms[i][second_part]))
		prev_transform = _inverse_transform(np.transpose(
			motion.rigid_transforms[i - 1][first_part])).dot(
			np.transpose(motion.rigid_transforms[i - 1][second_part]))

		cur_origin = cur_transform[0:3, 3]
		prev_origin = prev_transform[0:3, 3]
		cur_origin = motion.scale * np.array([cur_origin[0],cur_origin[2],cur_origin[1]])
		prev_origin = motion.scale * np.array([prev_origin[0],prev_origin[2],prev_origin[1]])

		origin_velocity = motion.time_step * (cur_origin - prev_origin)

		rotation = cur_transform[0:3, 0:3]

		angular_velocity_matrix = _log_rotation(rotation)

		angular_velocity = np.array([angular_velocity_matrix[2, 1],
		                             angular_velocity_matrix[0, 2],
		                             angular_velocity_matrix[1, 0]])

		twist = origin_velocity.tolist()
		twist += (origin_velocity + angular_velocity_matrix.dot(cur_origin)).tolist()

		twists += [twist]

	return twists


def n_squared_twist_from_rigid_trajectories(motion, use_parallel=False):
	if len(motion.rigid_parts_names) == 0 or len(motion.rigid_transforms) == 0:
		return []

	twists_dict = dict()
	job_packets = []
	for part_one in motion.rigid_parts_names:
		for part_two in motion.rigid_parts_names:
			if part_one != part_two:
				job_packets.append((motion, (part_one, part_two)))

	if use_parallel:
		from multiprocessing import Pool as ThreadPool
		pool = ThreadPool(None)
		twists = pool.map(n_squared_twist_from_rigid_trajectory, job_packets)
	else:
		twists = []
		for job in job_packets:
			twists.append(n_squared_twist_from_rigid_trajectory(job))

	for i in range(len(job_packets)):
		(motion, (part_one, part_two)) = job_packets[i]
		twists_dict[part_one+'_'+part_two] = twists[i]

	return twists_dict


def _log_rotation(rotation):

	skew_symmetric_direction = (rotation - np.transpose(rotation))
	rotation_norm = np.linalg.norm(skew_symmetric_direction)

	return (1.0/2.0*rotation_norm*np.sin(rotation_norm)) * skew_symmetric_direction


def _inverse_transform(transform):

	inverse_transform = np.zeros((4,4))

	inverse_transform[0:3, 0:3] = np.transpose(transform[0:3, 0:3])

	inverse_transform[0:3, 3] = - np.dot(inverse_transform[0:3, 0:3], transform[0:3, 3])

	return inverse_transform