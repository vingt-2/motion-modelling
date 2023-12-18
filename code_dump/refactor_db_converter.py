import os, time
from utils import qv_mult
import cPickle as pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
from PySide import QtGui, QtCore
import pyqtgraph as pg
from scipy import interpolate
from NatNet_acquisition import NatNetAcquisitionWidget

CAMERA_ID = 1
OBJECT_ID = 2

DB_FILE_EXTENSION = ".modb"

TRAJECTORY_NAME = "non-filtered-dist"
KF_NAME =  "keyframes"
ORIGIN_NAME = "original-data"
WARPED_MOTION = "warped-motion"
WARPED_FUNCTION = "warped-function"

class TimeWarp(object):

	def __init__(self):
		self.warp_spline = None
		self.evaluated_warp = []

	def compute_warp(self, ref_keyframes, keyframes):
		if len(keyframes) != len(ref_keyframes):
			return False

		spline = interpolate.PchipInterpolator(ref_keyframes, keyframes, axis=0)
		self.warp_spline = spline

		self.evaluated_function = [float(spline(x)) for x in range(ref_keyframes[0], ref_keyframes[-1])]

	def warp_trajectory(self, motion):
		warped_motion = []
		for t in self.evaluated_warp:
			f = int(np.floor(t))
			c = int(np.ceil(t))
			vf = motion[f]
			vc = motion[c]
			warped_motion.append(float(np.interp([t], [f, c], [vf, vc])))

class Motion(object):

	def __init__(self, name):
		self.print_c = None

		self.name = name
		self.original_trajectory = []
		self.edited_traj = []
		self.key_frames = []
		self.warped_motion = []
		self.time_warp = None

class MotionDatabase(object):

	def __init__(self):
		self.print_c = None
		
		self.dbFilename = ""
		self.reference_motion = "None"
		self.motions = []

	def add_motion(self, name, motion_traj):
		new_traj = {"name": name}
		new_traj.edited_traj = motion_traj
		new_traj.original_trajectory = motion_traj
		new_traj.warped_motion = []

		self.motions.append(new_traj)

	def load(self, data_path, filenames=None):

		(frame_start, frame_end) = (None, None) #Not supported for now... well see

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

				new_motion = Motion(filename.replace(".pickle",""))
				try:

					for frame in frames:
						for rb in frame.RigidBodies:
							if rb.id == CAMERA_ID:
								quat = (rb.qw, rb.qx, rb.qy, rb.qz)
								cam_dir = qv_mult(quat, (1,0,0))
								cam_pos = np.array([rb.x, rb.y, rb.z])
							if rb.id == OBJECT_ID:
								nofilt_pos = np.array([rb.x, rb.y, rb.z])

						latency_dist = np.dot(nofilt_pos - cam_pos, cam_dir)

						new_motion.edited_traj.append(latency_dist)
						new_motion.original_trajectory.append(latency_dist)

				except ValueError:
					self.print_c("Error in frame format... Probably wrong rigidbody count, or even no rigidbody !?")
					continue

				self.motions.append(new_motion)

class Motions(object):

	def __init__(self):
		self.trajs = []
		self.dbFilename = ""
		self.reference_motion = "None"
		self.print_c = None

	def add_traj(self, name, non_filt_data):
		new_traj = {"name": name}
		new_traj[TRAJECTORY_NAME] = non_filt_data
		new_traj[ORIGIN_NAME] = non_filt_data
		new_traj[WARPED_MOTION] = []
		new_traj[WARPED_FUNCTION] = []
		new_traj[KF_NAME] = []
		new_traj["eigen-motions"] = []
		new_traj["eigen-warps"] = []

		self.trajs.append(new_traj)


	def load(self, data_path, filenames=None):

		(frame_start, frame_end) = (None, None) #Not supported for now... well see

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

				new_traj = { "name" : filename.replace(".pickle","") }
				new_traj[TRAJECTORY_NAME] = []
				new_traj[ORIGIN_NAME] = []
				new_traj[KF_NAME] = []
				try:

					for frame in frames:
						for rb in frame.RigidBodies:
							if rb.id == CAMERA_ID:
								quat = (rb.qw, rb.qx, rb.qy, rb.qz)
								cam_dir = qv_mult(quat, (1,0,0))
								cam_pos = np.array([rb.x, rb.y, rb.z])
							if rb.id == OBJECT_ID:
								nofilt_pos = np.array([rb.x, rb.y, rb.z])

						latency_dist = np.dot(nofilt_pos - cam_pos, cam_dir)

						new_traj[TRAJECTORY_NAME].append(latency_dist)
						new_traj[ORIGIN_NAME].append(latency_dist)

				except ValueError:
					self.print_c("Error in frame format... Probably wrong rigidbody count, or even no rigidbody !?")
					continue

				self.trajs.append(new_traj)


def main():
	filename = "123456.modb"
	print ("Loading DB: " + filename)

	old_object = Motions()

	with open(filename, 'r') as fp:
		old_object = pickle.load(fp)

	new_object = MotionDatabase()
	new_object.dbFilename = old_object.dbFilename
	for traj in old_object.trajs:
		new_motion = Motion(traj["name"])
		new_motion.original_trajectory = traj[ORIGIN_NAME]
		new_motion.edited_traj = traj[TRAJECTORY_NAME]
		new_motion.key_frames = traj[KF_NAME]

		new_object.motions.append(new_motion)

	with open("converted_"+filename,'w') as fp:
		pickle.dump(new_object,fp)


if __name__ == "__main__":
	main()
