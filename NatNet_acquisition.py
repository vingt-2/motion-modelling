from __future__ import division

from collections import namedtuple
from pykalman import KalmanFilter
import cPickle as pickle
from utils import qv_mult
import transformations
import numpy as np
import NatNet
import scipy
import time

from PySide import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as pggl

Point = namedtuple('Point', ['x', 'y', 'z'])

forward = np.array([0.0, 0.0, 1.0])
frame = np.array([
	[ 0.1, 0, 0, 1],
	[ 0, 0.1, 0, 1],
	[ 0, 0, 0.1, 1],
	[-0.1, -0.1, -0.1, 1],
]).T

forward_L = (1.0 ,0.0 ,0.0)	## ! Important: Make sure this actually matches the Quat representation from NatNet

class PointTracker(object):
	'''A kalman tracked point'''
	def __init__(self):
		self.init = False

	def update(self, x, y, z):
		self.pos = np.array([x,y,z])

	def get_state(self):
		return Point(*self.pos)

class ObjectTracker(object):
	'''A kalman Object composed of multiple points'''
	def __init__(self):
		self.n = frame.shape[1]
		self.points = [PointTracker() for _ in range(self.n)]
		self.forward_W = forward_L

	def update(self, pos, quat):
		T = transformations.translation_matrix(pos)
		R = transformations.quaternion_matrix(quat)
		Ps = np.dot(np.dot(T, R), frame)

		self.forward_W = qv_mult(quat, forward_L)

		for i in range(self.n):
			self.points[i].update(*Ps[:3, i])

		new_Ps = [p.get_state() for p in self.points]

		c = np.matrix([
			sum(p.x for p in new_Ps ) / self.n,
			sum(p.y for p in new_Ps ) / self.n,
			sum(p.z for p in new_Ps ) / self.n,
			]).T

		A = np.dot(Ps[:3] - c, frame[:3].T)
		R, _ = scipy.linalg.polar(A)

		self.pos = np.squeeze(np.asarray(c))


class NatNetAcquisition(object):
	CAMERA_ID = 1
	OBJECT_ID = 2

	MIN_DIFF = 0.001

	def __init__(self):

		self.receive_data = True

		self.counter = 0
		self.rigidbodies = []
		self.offset = 0

		self.sampling_time = 0.0
		self.last_upd_time = 0.0
		self.last_upd_framecount = 0.0

		# Scene informationr
		self.obj = ObjectTracker()
		self.cam = ObjectTracker()

		self.forward_dist = 0

		self.record_frames = False

		self.obj_positions = []
		self.cam_positions = []

		# Initialize NatNet
		self.frames = []

		self.elapsed_time = 0
		self.timer_receive = 0

	def get_rigidbody_names(self):
		return self.rigidbodies

	def natnet_data(self, frame):
		sampling_rate_refresh_time = 2.0
		current_time = time.time()

		if current_time - self.last_upd_time > sampling_rate_refresh_time:

			elapsed_frames = len(self.frames) - self.last_upd_framecount

			self.sampling_time = 0 if (elapsed_frames == 0) else (sampling_rate_refresh_time / elapsed_frames)
			self.last_upd_framecount = len(self.frames)
			self.last_upd_time = current_time

		if self.receive_data:
			if self.elapsed_time > 4:
				self.rigidbodies = []
				if self.record_frames:
					self.frames.append(frame)
				for rb in frame.RigidBodies:
					self.rigidbodies.append(rb.id)
					if rb.id == self.CAMERA_ID:
						self.update_cam(rb)
					if rb.id == self.OBJECT_ID:
						self.update_obj(rb)
			else:
				if self.elapsed_time == 0:
					self.timer_receive = time.time()
					self.elapsed_time = 1
				else:
					self.elapsed_time = time.time() - self.timer_receive


			self.update_dist()
		else:
			self.elapsed_time = 0
			self.timer_receive = 0
	def update_obj(self, rb):
		quat = (rb.qw, rb.qx, rb.qy, rb.qz)
		pos = np.array([rb.x, rb.y, rb.z])
		self.obj_positions.append(pos)
		self.obj.update(pos, quat)

	def update_cam(self, rb):
		quat = (rb.qw, rb.qx, rb.qy, rb.qz)
		pos = np.array([rb.x, rb.y, rb.z])
		self.cam_positions.append(pos)
		self.cam.update(pos, quat)

	def update_dist(self):
		p1 = self.cam.pos
		p2 = self.obj.pos

		self.forward_dist = np.dot(p2 - p1, self.cam.forward_W)

		self.counter += 1

		self.plot.append(self.forward_dist)

class NatNetAcquisitionWidget(QtGui.QWidget):

	def __init__(self):

		super(NatNetAcquisitionWidget, self).__init__()

		self.col = -1

		self.last_packet_t = 0

		self.natnet = None

		self.print_c = None

		self.isAcquiring = False
		self.shouldExit = False
		self.shouldImport = False

		self.acquisition = NatNetAcquisition()

		self.ipField = QtGui.QLineEdit('192.168.52.28')
		self.connectStatText = QtGui.QLineEdit('Connection: None')
		self.connectStatText.setReadOnly(True)
		self.connectButton = QtGui.QPushButton('Connect')
		self.connectButton.clicked.connect(self.connect)

		self.doneButton = QtGui.QPushButton('Done')
		self.doneButton.clicked.connect(self.done)
		self.importButton = QtGui.QPushButton('<< \'n\' Done')
		self.importButton.clicked.connect(self.set_import)
		self.recordButton = QtGui.QPushButton('Start R')
		self.recordButton.clicked.connect(self.start_record)

		self.plot_view = pggl.GLViewWidget()
		self.object_traj_plot = pggl.GLLinePlotItem(pos=np.array([[0,0,0]]))

		self.plot_view.addItem(self.object_traj_plot)

		xgrid = pggl.GLGridItem()
		ygrid = pggl.GLGridItem()
		zgrid = pggl.GLGridItem()
		self.plot_view.addItem(xgrid)
		self.plot_view.addItem(ygrid)
		self.plot_view.addItem(zgrid)

		self.layout = QtGui.QGridLayout()
		self.setLayout(self.layout)

		self.layout.addWidget(self.doneButton, 0, 0)
		self.layout.addWidget(self.importButton, 1, 0)
		self.layout.addWidget(self.recordButton, 2, 0)
		self.layout.addWidget(self.plot_view, 2, 1, -1, 2)
		self.layout.addWidget(self.ipField, 0, 1)
		self.layout.addWidget(self.connectButton, 0, 2)
		self.layout.addWidget(self.connectStatText, 1, 1, 1, 2)
		verticalSpacer = QtGui.QSpacerItem(100, 1000, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
		self.layout.addItem(verticalSpacer, 3, 0, QtCore.Qt.AlignTop)

	def connect(self):
		ip_str = self.ipField.text()

		ip_parse = ip_str.split(".")
		if "" in ip_parse:
			ip_parse.remove("")

		if len(ip_parse) != 4 or any([(int(a) > 255 or int(a) < 0) for a in ip_parse]):
			self.print_c("Invalid Ip !")
			return

		self.print_c("Connecting to " + ip_str)
		self.natnet = NatNet.NatNetClient(1)
		self.natnet_status = self.natnet.Initialize("", ip_str)
		self.natnet.SetDataCallback(self.nat_net_interface)

		time.sleep(0.5)

		if self.natnet_status != 0 or self.last_packet_t == 0.0:
			self.print_c("Connection to NatNet failed. Status: " + str(self.natnet_status))
			self.connectStatText.setText("Connection: Failed")
		else:
			self.print_c("Connected to NatNet server")
			self.connectButton.clicked.connect(self.disconnect)
			self.connectButton.setText("Disconnect")
			self.ipField.setDisabled(True)
			self.connectStatText.setText("Connection: Established")

	def disconnect(self):
		self.natnet = NatNet.NatNetClient(0)
		self.natnet_status = self.natnet.Initialize("", "0")
		self.natnet.SetDataCallback(None)
		self.connectButton.clicked.connect(self.connect)
		self.connectButton.setText("Connect")
		self.ipField.setDisabled(False)

	def done(self):
		self.shouldExit = True
		if self.natnet != None:
			self.natnet.SetDataCallback(None)

	def set_import(self):
		self.shouldImport = True
		#self.shouldExit = True

	def start_record(self):
		self.isAcquiring = True
		self.recordButton.setText("Stop R")
		self.recordButton.clicked.connect(self.stop_record)

	def stop_record(self):
		self.isAcquiring = False
		self.recordButton.setText("Start R")
		self.recordButton.clicked.connect(self.start_record)

	def nat_net_interface(self, frame):
		self.last_packet_t = time.time()
		if self.isAcquiring:
			self.acquisition.natnet_data(frame)

	def update_plot(self):
		if len(self.acquisition.obj_positions) > 0:
			positions = [[m[0],m[2],m[1]] for m in self.acquisition.obj_positions]
			self.object_traj_plot.setData(pos=np.array(positions))