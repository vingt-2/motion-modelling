import os, time
from utils import qv_mult
import pickle
import numpy as np
from PySide6 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as pggl
from scipy import interpolate, optimize, stats, signal
#from NatNet_acquisition import NatNetAcquisitionWidget
from pykalman import KalmanFilter
from Motion_Database import Motion3DDatabase, Motion3D
from Warping_Registration import OnlineDynamicTimeWarp, GreedyWarp, SaliencyTimeWarp, _static_warp_motion
from sklearn.decomposition import PCA
from Motion_Database import twist_from_rigid_trajectories

DB_FILE_EXTENSION = ".mo3db"


def _static_non_blocking_wait(print_console_callback, t):
	ctime = time.time()
	while time.time() - ctime < t:
		print_console_callback("")


def _draw_skeleton_to_plot(plot, plot_items, motion, time_pos, colors=[1,1,1,1]):
	if motion.original_file_format == '.bvh' and len(motion.bones) != 0:
		if time_pos >= len(motion.trajectory):
			time_pos = len(motion.trajectory) - 1
		bones_at_t = motion.trajectory[time_pos]
		i = 0
		for bone in motion.bones:
			i_joint_0 = motion.markers.index(bone[0])
			i_joint_1 = motion.markers.index(bone[1])

			joint_1_pose = bones_at_t[3 * i_joint_0:3 * i_joint_0 + 3]
			joint_2_pose = bones_at_t[3 * i_joint_1:3 * i_joint_1 + 3]

			bone_pose = [joint_1_pose, joint_2_pose]

			bone_center = 0.5 * (np.array(bone_pose[0]) + np.array(bone_pose[1]))

			bone_plot = pggl.GLLinePlotItem(pos=np.array(bone_pose), color=np.array([colors, colors]))

			plot.addItem(bone_plot)
			plot_items.append(bone_plot)

			i += 1


def _draw_skeleton_transforms_to_plot(plot, plot_items, motion, selected_marker, time_pos):
	if motion.original_file_format == '.bvh' and len(motion.rigid_parts_names) != 0:
		for part_name in motion.rigid_parts_names:
			transform = np.transpose(motion.rigid_transforms[time_pos][part_name])

			origin = transform.dot(np.array([0, 0, 0, 1]))

			origin = motion.scale * np.array([origin[0], origin[2], origin[1]])

			up = transform.dot(np.array([0, 1, 0, 0]))[:3]
			left = transform.dot(np.array([0, 0, 1, 0]))[:3]
			forward = transform.dot(np.array([1, 0, 0, 0]))[:3]

			up_disp = [origin, origin + up / (5*np.linalg.norm(up))]
			left_disp = [origin, origin + left / (5*np.linalg.norm(left))]
			forward_disp = [origin, origin + forward / (5*np.linalg.norm(forward))]

			up_plot = pggl.GLLinePlotItem(pos=np.array(up_disp), color=np.array([[0, 1, 0, 1], [0, 1, 0, 1]]))
			left_plot = pggl.GLLinePlotItem(pos=np.array(left_disp), color=np.array([[0, 0, 1, 1], [0, 0, 1, 1]]))
			forward_plot = pggl.GLLinePlotItem(pos=np.array(forward_disp), color=np.array([[1, 0, 0, 1], [1, 0, 0, 1]]))

			plot.addItem(up_plot)
			plot_items.append(up_plot)

			plot.addItem(left_plot)
			plot_items.append(left_plot)

			plot.addItem(forward_plot)
			plot_items.append(forward_plot)

			if selected_marker == part_name:

				from Motion_Database import _inverse_transform, _log_rotation

				for second_part in motion.rigid_parts_names:
					if part_name != second_part:
						cur_transform = _inverse_transform(np.transpose(
							motion.rigid_transforms[time_pos][part_name])).dot(
							np.transpose(motion.rigid_transforms[time_pos][second_part]))
						prev_transform = _inverse_transform(np.transpose(
							motion.rigid_transforms[time_pos - 1][part_name])).dot(
							np.transpose(motion.rigid_transforms[time_pos - 1][second_part]))

						cur_origin = cur_transform[0:3, 3]
						prev_origin = prev_transform[0:3, 3]
						cur_origin = motion.scale * np.array([cur_origin[0], cur_origin[2], cur_origin[1]])
						prev_origin = motion.scale * np.array([prev_origin[0], prev_origin[2], prev_origin[1]])

						origin_velocity = motion.time_step * (cur_origin - prev_origin)

						rotation = cur_transform[0:3, 0:3]

						angular_velocity_matrix = _log_rotation(rotation)

						linear_velocity = origin_velocity
						angular_velocity = origin_velocity + angular_velocity_matrix.dot(cur_origin)

						up = cur_transform.dot(np.array([0, 1, 0, 0]))[:3]
						left = cur_transform.dot(np.array([0, 0, 1, 0]))[:3]
						forward = cur_transform.dot(np.array([1, 0, 0, 0]))[:3]

						up_disp = [cur_origin, cur_origin + up / (5 * np.linalg.norm(up))]
						left_disp = [cur_origin, cur_origin + left / (5 * np.linalg.norm(left))]
						forward_disp = [cur_origin, cur_origin + forward / (5 * np.linalg.norm(forward))]

						up_plot = pggl.GLLinePlotItem(pos=np.array(up_disp),
						                              color=np.array(
							                              [[1., 0.41, 0.70, 1], [1., 0.41, 0.70, 1]]))
						left_plot = pggl.GLLinePlotItem(pos=np.array(left_disp),
						                                color=np.array(
							                                [[1., 0.41, 0.70, 1], [1., 0.41, 0.70, 1]]))
						forward_plot = pggl.GLLinePlotItem(pos=np.array(forward_disp),
						                                   color=np.array(
							                                   [[1., 0.41, 0.70, 1], [1., 0.41, 0.70, 1]]))

						linear_velocity_disp = [cur_origin.tolist(), (cur_origin + linear_velocity).tolist()]
						angular_velocity_disp = [cur_origin.tolist(), (cur_origin + angular_velocity).tolist()]

						linear_vel_plot = pggl.GLLinePlotItem(pos=np.array(linear_velocity_disp),
						                                      color=np.array([[1, 0.4, 0, 1], [1, 0.4, 0, 1]]))
						angular_vel_plot = pggl.GLLinePlotItem(pos=np.array(angular_velocity_disp),
						                                       color=np.array([[0.8, 0, 1, 1], [0.8, 0, 1, 1]]))

						plot.addItem(linear_vel_plot)
						plot_items.append(linear_vel_plot)

						plot.addItem(angular_vel_plot)
						plot_items.append(angular_vel_plot)

						plot.addItem(up_plot)
						plot_items.append(up_plot)

						plot.addItem(left_plot)
						plot_items.append(left_plot)

						plot.addItem(forward_plot)
						plot_items.append(forward_plot)


def _draw_markers_to_plot(plot, plot_items, motion, time_pos, trail_length, colors=[1 ,1, 1, 1]):

	time_colors = [[colors[0], colors[1], colors[2], float(a) / trail_length] \
	               for a in range(0, int(trail_length))]

	for marker_index in range(0, len(motion.markers)):
		marker_trajectory = [p[3 * marker_index:3 * marker_index + 3] for p in \
		                     motion.trajectory[time_pos - trail_length:time_pos]]

		current_marker_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory),
			                                          color=np.array(time_colors))

		plot.addItem(current_marker_plot)
		plot_items.append(current_marker_plot)


def _draw_markers_to_plot_highlight_marker(plot, plot_items, motion, selected_markers, time_pos, trail_length):

	time_colors = [3 * [float(a) / trail_length] \
	               for a in range(0, int(trail_length))]

	selected_markers_colors = [[1, 0, 0, float(a) / trail_length] \
	                           for a in range(0, int(trail_length))]

	for marker_index in range(0, len(motion.markers)):

		marker_name = motion.markers[marker_index]

		marker_trajectory = [p[3 * marker_index:3 * marker_index + 3] for p in \
		                     motion.trajectory[time_pos - trail_length:time_pos]]

		if selected_markers == marker_name:
			current_marker_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory),
			                                          color=np.array(selected_markers_colors))
		else:
			current_marker_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory),
			                                          color=np.array(time_colors))

		plot.addItem(current_marker_plot)
		plot_items.append(current_marker_plot)


def _clear_plot(plot, plot_items):
	for item in plot_items:
		plot.removeItem(item)

	plot_items.clear()


class MotionEditionWidget(QtWidgets.QWidget):
	def __init__(self):

		super(MotionEditionWidget, self).__init__()

		self.twists = []

		self.spacePressed = False
		self.keyFramesPositions = []
		self.col = -1
		self.synth_count = 0
		self.print_c = None

		self.motions_db = Motion3DDatabase()

		self.derivative_mode = 0

		self.plot_display = "markers"

		self.motion_name_list = QtWidgets.QListWidget()
		self.motion_name_list.itemChanged.connect(self.motion_name_edit)
		self.motion_name_list.currentItemChanged.connect(self.list_change_event)

		self.markers_list = QtWidgets.QListWidget()
		self.markers_list.currentItemChanged.connect(self.markers_list_change_event)

		self.persp_plot = pggl.GLViewWidget()
		self.current_plot_items = []

		self.trail_length_box = QtWidgets.QSpinBox()
		self.trail_length_box.setValue(20)
		self.trail_length_box.setMinimum(1)
		self.trail_length_box.valueChanged.connect(self.trail_length_box_changed)

		xgrid = pggl.GLGridItem()
		ygrid = pggl.GLGridItem()
		zgrid = pggl.GLGridItem()
		self.persp_plot.addItem(xgrid)
		self.persp_plot.addItem(ygrid)
		self.persp_plot.addItem(zgrid)

		self.time_slider = QtWidgets.QSlider()
		self.time_slider.setOrientation(QtCore.Qt.Orientation.Vertical)
		self.time_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksAbove)
		self.time_slider.valueChanged.connect(self.slider_moved)

		self.time_box = QtWidgets.QSpinBox()
		self.time_box.setValue(0)
		self.time_box.setMinimum(0)
		self.time_box.setMaximum(0)
		self.time_box.valueChanged.connect(self.time_box_event)

		self.time_series_plot = pg.PlotWidget()
		self.time_series_plot_legend = self.time_series_plot.addLegend()
		self.time_series_frame_line = self.time_series_plot.addLine(0)

		self.layout = QtWidgets.QGridLayout()

		self.show_bones_buttons = QtWidgets.QPushButton('Show Bones')
		self.show_bones_buttons.clicked.connect(self.show_bones)

		self.derivative_button = QtWidgets.QPushButton('Delete Motion')
		self.derivative_button.clicked.connect(self.delete_motion_event)

		self.remove_markers_button = QtWidgets.QPushButton('Only Keep common markers')
		self.remove_markers_button.clicked.connect(self.remove_markers)

		self.clamp_s_button = QtWidgets.QPushButton('Clamp start')
		self.clamp_s_button.clicked.connect(self.clamp_start)

		self.clamp_e_button = QtWidgets.QPushButton('Clamp end')
		self.clamp_e_button.clicked.connect(self.clamp_end)

		self.kf_button = QtWidgets.QPushButton('Add Keyframe')
		self.kf_button.clicked.connect(self.add_kf_event)

		self.split_motions_button = QtWidgets.QPushButton('Split Motions')
		self.split_motions_button.clicked.connect(self.split_motions_event)

		self.spit_out_twist_button = QtWidgets.QPushButton('Dump n^squared Twists')
		self.spit_out_twist_button.clicked.connect(self.write_twist_trajectories)

		self.dump_markers_trajectories = QtWidgets.QPushButton('Dump markers Trajectories')
		self.dump_markers_trajectories.clicked.connect(self.write_markers_trajectories)

		self.layout.addWidget(QtWidgets.QLabel("Motions: "), 0, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.motion_name_list, 1, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.persp_plot, 0, 1, -1, 3)
		self.layout.addWidget(self.time_slider, 0, 4, -1, 1)
		self.layout.addWidget(self.time_box, 1, 4, 1, 1)
		self.layout.addWidget(self.time_series_plot,0, 5, -1, 2, QtCore.Qt.AlignmentFlag.AlignTop)

		self.layout.addWidget(self.show_bones_buttons, 4, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.derivative_button, 5, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.kf_button, 6, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.clamp_s_button, 7, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.clamp_e_button, 8, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.remove_markers_button, 9, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(QtWidgets.QLabel("Trail_Size: "), 10, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.trail_length_box, 11, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(QtWidgets.QLabel("Markers: "), 12, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.markers_list, 13, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.split_motions_button, 14, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.spit_out_twist_button, 15, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.dump_markers_trajectories, 16, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		verticalSpacer = QtWidgets.QSpacerItem(100, 1000, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
		self.layout.addItem(verticalSpacer, 17, 0, QtCore.Qt.AlignmentFlag.AlignTop)

		self.setLayout(self.layout)

		self.refresh_all()

	def time_box_event(self):
		self.time_slider.setValue(self.time_box.value())
		self.time_series_plot.removeItem(self.time_series_frame_line)
		self.time_series_frame_line = self.time_series_plot.addLine(self.time_slider.value())
		self.refresh_persp_plot()

	def add_kf_event(self):
		self.keyFramesPositions += [self.time_slider.value()]
		self.time_series_plot.addLine(self.time_slider.value())

	def split_motions_event(self):
		motion = self.motions_db.motions[self.motion_name_list.currentRow()]
		cur_time = time.time()

		if(len(self.keyFramesPositions) < 2):
			return

		for i in range(len(self.keyFramesPositions)-1):
			if self.keyFramesPositions[i+1] >= len(motion.trajectory):
				break

			new_motion = Motion3D(motion.name + "_cut_t=" + str(cur_time)[-3:] + "_" + str(i))
			new_motion.original_file_format = motion.original_file_format
			new_motion.original_trajectory = motion.trajectory[self.keyFramesPositions[i]:self.keyFramesPositions[i+1]]
			new_motion.trajectory = new_motion.original_trajectory
			new_motion.markers = motion.markers
			new_motion.bones = motion.bones
			new_motion.scale = motion.scale

			new_motion.rigid_parts_names = motion.rigid_parts_names
			new_motion.rigid_transforms = motion.rigid_transforms[self.keyFramesPositions[i]:self.keyFramesPositions[i+1]]

			for part_to_part_twist in motion.n_squared_twists_trajectories:
				new_motion.n_squared_twists_trajectories[part_to_part_twist] = motion.n_squared_twists_trajectories\
															[part_to_part_twist]\
															[self.keyFramesPositions[i]:self.keyFramesPositions[i+1]]

			self.motions_db.motions.append(new_motion)

		self.refresh_all()

	def keyPressEvent(self, event):
		#super(MotionEditionWidget, self).keyPressEvent(event)
		self.on_space_key()

	def on_space_key(self):
		self.spacePressed = True

	def wait_for_space(self):
		while not self.spacePressed:
			self.print_c("")
		self.spacePressed = False

	def slider_moved(self):
		self.time_series_plot.removeItem(self.time_series_frame_line)
		self.time_series_frame_line = self.time_series_plot.addLine(self.time_slider.value())
		self.time_box.setValue(self.time_slider.value())
		self.refresh_persp_plot()

	def delete_motion_event(self):
		index = self.motion_name_list.currentRow()
		if index > -1:
			self.motions_db.motions.remove(self.motions_db.motions[index])
			self.refresh_all()

	def remove_markers(self):
		common_markers = set(self.motions_db.motions[0].markers)
		for i in range(1,len(self.motions_db.motions)):
			common_markers = common_markers & set(self.motions_db.motions[i].markers)

		for motion in self.motions_db.motions:
			transposed_traj = []
			new_markers = []

			dims = np.transpose(motion.trajectory)
			for m in common_markers:
				index = -1
				for i in range(0,len(motion.markers)):
					if m == motion.markers[i]:
						index = i

				if index != -1:
					transposed_traj.extend(dims[3 * index : 3 * (index + 1)])
					new_markers.append(motion.markers[index])

			motion.trajectory = np.transpose(transposed_traj)
			motion.markers = new_markers

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

			if self.derivative_mode == 0:
				motion.trajectory = motion.original_trajectory
				continue

			motion.trajectory = motion.differentiate(self.derivative_mode, True)

		self.refresh_persp_plot()

	def clamp_start(self):
		index = self.motion_name_list.currentRow()
		if index > -1:
			i = self.time_slider.value()
			self.motions_db.motions[index].trajectory = self.motions_db.motions[index].trajectory[i:]
			self.motions_db.motions[index].original_trajectory = self.motions_db.motions[index].trajectory

		self.refresh_all()

	def clamp_end(self):
		index = self.motion_name_list.currentRow()
		if index > -1:
			i = self.time_slider.value()
			self.motions_db.motions[index].trajectory = self.motions_db.motions[index].trajectory[:i]
			self.motions_db.motions[index].original_trajectory = self.motions_db.motions[index].trajectory
		self.refresh_all()

	def refresh_persp_plot(self):

		_clear_plot(self.persp_plot, self.current_plot_items)

		motion_index = self.motion_name_list.currentRow()
		selected_marker_index = self.markers_list.currentRow()
		if motion_index > -1:
			motion = self.motions_db.motions[motion_index]

			trail_length = self.trail_length_box.value()

			time_colors = [3*[float(a)/trail_length]\
			               for a in range(0,int(trail_length))]

			selected_markers_colors = [[1,0,0,float(a)/trail_length]\
			               for a in range(0,int(trail_length))]

			time_slider_pos = self.time_slider.value()

			if self.plot_display == "bones":
				_draw_skeleton_to_plot(self.persp_plot, self.current_plot_items, motion, time_slider_pos)

			elif self.plot_display == "transforms":
				_draw_skeleton_transforms_to_plot(self.persp_plot, self.current_plot_items, motion, motion.markers[selected_marker_index], time_slider_pos)
			else:
				_draw_markers_to_plot_highlight_marker(self.persp_plot, self.current_plot_items, motion, motion.markers[selected_marker_index], time_slider_pos, trail_length)

	def refresh_motion_name_list(self):
		i = 0
		self.motion_name_list.clear()

		for t in self.motions_db.motions:
			i += 1
			li = QtWidgets.QListWidgetItem()
			li.setFlags(li.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
			li.setText(t.name)
			self.motion_name_list.insertItem(i, li)

		self.motion_name_list.setMaximumWidth(200)

	def refresh_all(self):
		self.refresh_motion_name_list()
		self.refresh_markers_list()
		self.refresh_persp_plot()

	def show_bones(self):
		if self.plot_display == "markers":
			self.plot_display = "bones"
			self.show_bones_buttons.setText("Show Transforms")
			self.refresh_persp_plot()
			return

		if self.plot_display == "bones":
			self.plot_display = "transforms"
			self.show_bones_buttons.setText("Show Markers")
			self.refresh_persp_plot()
			return

		if self.plot_display == "transforms":
			self.plot_display = "markers"
			self.show_bones_buttons.setText("Show Bones")
			self.refresh_persp_plot()
			return

	def list_change_event(self):
		index = self.motion_name_list.currentRow()

		if index > -1:
			motion = self.motions_db.motions[index]

			self.show_bones_buttons.setDisabled(len(motion.bones) < 1)

			self.time_slider.setMaximum(len(motion.trajectory))
			self.time_box.setMaximum(len(motion.trajectory))
			self.trail_length_box.setMaximum(len(motion.trajectory))
			self.trail_length_box_changed()

			self.refresh_markers_list()

			self.twists = twist_from_rigid_trajectories(motion)

		self.refresh_persp_plot()

	def trail_length_box_changed(self):
		self.time_slider.setMinimum(self.trail_length_box.value())
		self.refresh_persp_plot()

	def markers_list_change_event(self):
		motion_index = self.motion_name_list.currentRow()
		marker_index = self.markers_list.currentRow()
		if motion_index > -1:
			motion = self.motions_db.motions[motion_index]

			self.time_series_plot.clear()
			self.time_series_plot.removeItem(self.time_series_plot_legend)
			self.time_series_plot_legend.close()
			self.time_series_plot_legend = self.time_series_plot.addLegend()

			marker_trajectory_x = [p[3 * marker_index + 0] for p in motion.trajectory]
			marker_trajectory_y = [p[3 * marker_index + 1] for p in motion.trajectory]
			marker_trajectory_z = [p[3 * marker_index + 2] for p in motion.trajectory]

			self.time_series_plot.plot(np.array(marker_trajectory_x), pen=(0, 3), name="X")
			self.time_series_plot.plot(np.array(marker_trajectory_y), pen=(1, 3), name="Y")
			self.time_series_plot.plot(np.array(marker_trajectory_z), pen=(2, 3), name="Z")
		self.refresh_persp_plot()

	def refresh_markers_list(self):
		index = self.motion_name_list.currentRow()
		if index > -1:
			motion = self.motions_db.motions[index]
			i = 0
			self.markers_list.clear()
			for m_name in motion.markers:
				i += 1
				li = QtWidgets.QListWidgetItem()
				li.setText(m_name)
				self.markers_list.insertItem(i, li)

		self.markers_list.setMaximumWidth(200)
		self.refresh_persp_plot()

	def motion_name_edit(self, list_item):
		name = list_item.text()

		item_index = self.motion_name_list.indexFromItem(list_item).row()

		if len(self.motion_name_list.findItems(name, 1)) > 1:
			self.refresh_motion_name_list()
			return

		self.motions_db.motions[item_index].name = name

		self.refresh_markers_list()

	def write_twist_trajectories(self):
		self.print_c("Spitting out the twist files for this db ...")
		for motion in self.motions_db.motions:
			if len(motion.n_squared_twists_trajectories) == 0:
				continue
			#twists = twist_from_rigid_trajectories(motion)
			twists = motion.n_squared_twists_trajectories
			trajectory = []
			for rigid_part in twists:
				part_traj = np.transpose(twists[rigid_part])

				for dim in part_traj:
					trajectory.append(dim)

			filename = "n^2_twists_features/n^2_twists_" + motion.name + ".csv"
			np.savetxt(filename, trajectory, delimiter=",")

			self.print_c("Created file: " + filename)

	def write_markers_trajectories(self):
		self.print_c("Spitting out the Marker trajectory files for this db ...")
		for motion in self.motions_db.motions:
			if len(motion.trajectory) == 0:
				continue

			dense_features = True

			trajectory = motion.trajectory

			if dense_features:
				densed_trajectory = []
				for i in range(0,len(trajectory)):
					dense_frame = []
					for j in range(int(len(trajectory[i])/3)):
						for h in range(int(len(trajectory[i])/3)):
							if j == h:
								continue
							dense_frame += (np.array(trajectory[i][3*j:3*j+3]) - np.array(trajectory[i][3*h:3*h+3])).tolist()
					densed_trajectory += [dense_frame]
				filename = "markers_features/dense_markers_" + motion.name + ".csv"
				np.savetxt(filename, densed_trajectory, delimiter=",")
			else:
				filename = "markers_features/markers_" + motion.name + ".csv"
				np.savetxt(filename, trajectory, delimiter=",")

			self.print_c("Created file: " + filename)


class TrajectoryAnalysisWidget(QtWidgets.QWidget):
	def __init__(self):

		super(TrajectoryAnalysisWidget, self).__init__()
		self.col = -1
		self.synth_count = 0
		self.print_c = None

		self.motions_db = Motion3DDatabase()

		self.projected_motions = []
		self.reconstructed_motions = []

		self.sklearn_pca = None

		self.derivative_mode = 0

		self.traj_list_ref = QtWidgets.QListWidget()
		self.traj_list_ref.currentItemChanged.connect(self.list_change_event)

		self.persp_plot = pggl.GLViewWidget()
		self.persp_plot_items = []

		self.trail_length_box = QtWidgets.QSpinBox()
		self.trail_length_box.setValue(20)
		self.trail_length_box.setMinimum(1)
		self.trail_length_box.valueChanged.connect(self.trail_length_box_changed)

		self.n_components_box = QtWidgets.QSpinBox()
		self.n_components_box.setValue(20)
		self.n_components_box.setMinimum(1)

		xgrid = pggl.GLGridItem()
		ygrid = pggl.GLGridItem()
		zgrid = pggl.GLGridItem()
		self.persp_plot.addItem(xgrid)
		self.persp_plot.addItem(ygrid)
		self.persp_plot.addItem(zgrid)

		self.time_slider = QtWidgets.QSlider()
		self.time_slider.setOrientation(QtCore.Qt.Orientation.Vertical)
		self.time_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksAbove)
		self.time_slider.valueChanged.connect(self.refresh_presp_plot)

		self.pcs_plot = pg.PlotWidget()
		self.pcs_plot.addLegend()

		self.layout = QtWidgets.QGridLayout()

		self.derivative_button = QtWidgets.QPushButton('Saliency')
		self.derivative_button.clicked.connect(self.saliency_matching_showcase)

		self.compute_pca_button = QtWidgets.QPushButton('Compute PCA space')
		self.compute_pca_button.clicked.connect(self.compute_pca_clicked)

		self.layout.addWidget(QtWidgets.QLabel("Motions: "), 0, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.traj_list_ref, 1, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.persp_plot, 0, 1, -1, 3)
		self.layout.addWidget(self.time_slider, 0, 4, -1, 1)
		self.layout.addWidget(self.pcs_plot, 0, 5, -1, 2)

		self.layout.addWidget(self.derivative_button, 4, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.compute_pca_button, 5, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(QtWidgets.QLabel("Trail_Size: "), 8, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.trail_length_box, 9, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(QtWidgets.QLabel("# components: "), 10, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.n_components_box, 11, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		verticalSpacer = QtWidgets.QSpacerItem(100, 1000, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
		self.layout.addItem(verticalSpacer, 12, 0, QtCore.Qt.AlignmentFlag.AlignTop)

		self.setLayout(self.layout)

		self.refresh_all()

	def saliency_matching_showcase(self):
		ref_indx = self.traj_list_ref.currentRow()
		self.registered_ref = ref_indx
		if ref_indx == -1:
			#self.print_c("No Reference Trajectory Selected")
			return

		ref_motion = self.motions_db.motions[ref_indx]

		pose_matrix = []

		for m in self.motions_db.motions:
			for p in m.trajectory:
				pose_matrix.append(p)

		pose_matrix = np.array(pose_matrix)

		trail_length = 10

		for i in range(0, len(pose_matrix)):
			mean_x = np.average([pose_matrix[i][j] for j in range(0, len(pose_matrix[i]), 3)])
			mean_y = np.average([pose_matrix[i][j] for j in range(1, len(pose_matrix[i]), 3)])
			mean_z = np.average([pose_matrix[i][j] for j in range(2, len(pose_matrix[i]), 3)])

			for j in range(0, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_x
			for j in range(1, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_y
			for j in range(2, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_z

		n_components = self.n_components_box.value()

		pca = PCA(n_components=n_components)
		pca.fit(np.array(pose_matrix))

		principal_components_traj = pca.transform(ref_motion.trajectory)

		saliency = SaliencyTimeWarp(principal_components_traj, show_graphs=True)

		self.print_c("Saliency: " + str(saliency.salient_points_ref))

		salient_points = [i for (j,i,pos) in saliency.salient_points_ref]

		draw_ref_colors = [[1,0,0,1] if len(set(range(p,p+1)) & set(salient_points))>0 else [1,1,1,0.05]\
		                   for p in range(trail_length,len(ref_motion.trajectory)-1)]

		for plot in self.persp_plot_items:
			self.persp_plot.removeItem(plot)
		self.persp_plot_items = []

		for marker_index in range(0, len(ref_motion.markers)):
			marker_trajectory = [p[3 * marker_index:3 * marker_index + 3] for p in \
			                     ref_motion.trajectory]

			current_marker_right_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory),
			                                                color=np.array(draw_ref_colors))
			self.persp_plot.addItem(current_marker_right_plot)
			self.persp_plot_items.append(current_marker_right_plot)

	def compute_pca_clicked(self):
		pose_matrix = []

		ref_indx = self.traj_list_ref.currentRow()
		ref_indx = ref_indx if ref_indx != -1 else 0

		for m_i in range(len(self.motions_db.motions)):
			if m_i != ref_indx:
				m = self.motions_db.motions[m_i]
				for p in m.trajectory:
					pose_matrix.append(p)

		pose_matrix = np.array(pose_matrix)

		for i in range(0, len(pose_matrix)):
			mean_x = np.average([pose_matrix[i][j] for j in range(0, len(pose_matrix[i]), 3)])
			mean_y = np.average([pose_matrix[i][j] for j in range(1, len(pose_matrix[i]), 3)])
			mean_z = np.average([pose_matrix[i][j] for j in range(2, len(pose_matrix[i]), 3)])

			for j in range(0, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_x
			for j in range(1, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_y
			for j in range(2, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_z


		n_components=self.n_components_box.value()


		pca = PCA(n_components=n_components)
		pca.fit(np.array(pose_matrix))

		self.sklearn_pca = pca

		explained_variance_plot = pg.plot(title="Explained Variance per Component")
		explained_variance_plot.plot(pca.explained_variance_)
		explained_variance_plot.show()

		self.reconstructed_motions = []
		for m in self.motions_db.motions:
			projected_motion = pca.transform(m.trajectory)
			reconstructed_motion = pca.inverse_transform(projected_motion)
			self.reconstructed_motions.append(reconstructed_motion)
			self.projected_motions.append(projected_motion)

		self.pcs_plot.clear()

		for i in range(n_components):
			self.pcs_plot.plot([(x[i]) for x in self.projected_motions[ref_indx]], pen=(i, n_components),
			                       name="C#" + str(i))

		# first_two_pcs_widget = pg.plot([x[0] for x in self.projected_motions[1]],[x[1] for x in self.projected_motions[1]])
		# first_two_pcs_widget.show()

		self.refresh_presp_plot()

	def derivative_clicked(self):
		self.refresh_presp_plot()

	def refresh_presp_plot(self):
		for plot in self.persp_plot_items:
			self.persp_plot.removeItem(plot)

		self.persp_plot_items = []
		index = self.traj_list_ref.currentRow()
		if index > -1:
			motion = self.motions_db.motions[index]

			trail_length = self.trail_length_box.value()

			time_colors = [3*[float(a)/trail_length]\
			               for a in range(0,int(trail_length))]

			time_slider_pos = self.time_slider.value()

			trajectory = motion.trajectory if self.reconstructed_motions == [] else self.reconstructed_motions[index]

			for marker_index in range(0,len(motion.markers)):
				marker_trajectory = [p[3*marker_index:3*marker_index+3] for p in \
				                     trajectory[time_slider_pos-trail_length:time_slider_pos]]

				current_marker_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory), color=np.array(time_colors))
				self.persp_plot.addItem(current_marker_plot)
				self.persp_plot_items.append(current_marker_plot)

	def refresh_traj_list(self):
		i = 0
		self.traj_list_ref.clear()
		for t in self.motions_db.motions:
			i += 1
			li = QtWidgets.QListWidgetItem()
			li.setText(t.name)
			self.traj_list_ref.insertItem(i, li)

		self.traj_list_ref.setMaximumWidth(200)

	def refresh_all(self):
		self.refresh_traj_list()
		self.refresh_presp_plot()

	def list_change_event(self):
		index = self.traj_list_ref.currentRow()
		if index > -1:
			motion = self.motions_db.motions[index]
			self.time_slider.setMaximum(len(motion.trajectory))
			self.trail_length_box.setMaximum(len(motion.trajectory))
			self.trail_length_box_changed()

		self.refresh_presp_plot()

	def trail_length_box_changed(self):
		self.time_slider.setMinimum(self.trail_length_box.value())
		self.refresh_presp_plot()


class TrajectoryPredictionWidget(QtWidgets.QWidget):
	def __init__(self):

		self.col = -1

		self.is_show_splines = False
		self.is_registering = False
		self.is_prediction_showcase = False

		self.registered_ref = -1

		super(TrajectoryPredictionWidget, self).__init__()

		self.print_c = None

		self.motions_db = Motion3DDatabase()

		self.spacePressed = False
		self.lastTimeSpacePressed = time.time()
		self.latest_computed_warp = []

		self.traj_list_ref = QtWidgets.QListWidget()
		self.traj_list_ref.currentItemChanged.connect(self.list_change_event)

		self.traj_list_reg = QtWidgets.QListWidget()
		self.traj_list_reg.currentItemChanged.connect(self.list_change_event)

		self.left_plot_view = pggl.GLViewWidget()
		self.left_plot_items = []
		self.right_plot_view = pggl.GLViewWidget()
		self.right_plot_items = []

		xgrid = pggl.GLGridItem()
		ygrid = pggl.GLGridItem()
		zgrid = pggl.GLGridItem()
		self.left_plot_view.addItem(xgrid)
		self.left_plot_view.addItem(ygrid)
		self.left_plot_view.addItem(zgrid)

		self.right_plot_view.addItem(xgrid)
		self.right_plot_view.addItem(ygrid)
		self.right_plot_view.addItem(zgrid)

		self.layout = QtWidgets.QGridLayout()

		self.dtw_reg_button = QtWidgets.QPushButton('DTW Warp')
		self.dtw_reg_button.pressed.connect(self.dtw_registration_showcase)#dtw_registration_showcasegreedy_registration_showcase)

		self.pred_one_one_button = QtWidgets.QPushButton('Prediction 1v1 showcase')
		self.pred_one_one_button.pressed.connect(self.prediction_one_on_one)

		self.new_test = QtWidgets.QPushButton('Dont click')
		#self.new_test.pressed.connect(self.saliency_matching_showcase)

		self.kal_pred_button = QtWidgets.QPushButton('Regular Kalman')
		self.kal_pred_button.pressed.connect(self.regular_kalman_prediction)

		self.time_slider = QtWidgets.QSlider()

		self.layout.addWidget(QtWidgets.QLabel("Reference Motions: "), 0, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.traj_list_ref, 1, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.pred_one_one_button, 5, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.new_test, 6, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.dtw_reg_button, 7, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.kal_pred_button, 8, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(QtWidgets.QLabel("Register Motion: "), 9, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.traj_list_reg, 10, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.left_plot_view, 0, 1, -1, 1)
		self.layout.addWidget(self.right_plot_view, 0, 2, -1, 1)

		verticalSpacer = QtWidgets.QSpacerItem(100, 3000, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
		self.layout.addItem(verticalSpacer, 9, 0, QtCore.Qt.AlignmentFlag.AlignTop)

		self.setLayout(self.layout)

		self.current_warp = None

		self.refresh_all()

	def keyPressEvent(self, event):
		#super(MotionEditionWidget, self).keyPressEvent(event)
		self.on_space_key()

	def on_space_key(self):
		if time.time() - self.lastTimeSpacePressed > 0.2:
			self.spacePressed = True
			self.lastTimeSpacePressed = time.time()

	def wait_for_space(self):
		while not self.spacePressed:
			self.print_c("")
		self.spacePressed = False

	def detect_space(self):
		if self.spacePressed:
			self.spacePressed = False
			return True
		return False

	def dtw_registration_showcase(self):

		ref_indx = self.traj_list_ref.currentRow()
		self.registered_ref = ref_indx
		if ref_indx == -1:
			self.print_c("No Reference Trajectory Selected")
			return

		reg_indx = self.traj_list_reg.currentRow()
		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		input_motion = self.motions_db.motions[reg_indx]
		ref_motion = self.motions_db.motions[ref_indx]

		pose_matrix = []

		for m in self.motions_db.motions:
			if m.name != input_motion.name:
				for p in m.trajectory:
					pose_matrix.append(p)

		pose_matrix = np.array(pose_matrix)

		for i in range(0, len(pose_matrix)):
			mean_x = np.average([pose_matrix[i][j] for j in range(0, len(pose_matrix[i]), 3)])
			mean_y = np.average([pose_matrix[i][j] for j in range(1, len(pose_matrix[i]), 3)])
			mean_z = np.average([pose_matrix[i][j] for j in range(2, len(pose_matrix[i]), 3)])

			for j in range(0, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_x
			for j in range(1, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_y
			for j in range(2, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_z

		n_components = 6

		pca = PCA(n_components=n_components)
		pca.fit(np.array(pose_matrix))

		pca_trajectory = pca.transform(ref_motion.trajectory)

		warping = SaliencyTimeWarp(pca_trajectory, show_graphs=False)

		arriving_pca = []

		trail_length = 3

		for i in range(len(input_motion.trajectory)):
			_static_non_blocking_wait(self.print_c, 0.00001)

			if self.detect_space():
				self.wait_for_space()

			pca_new_point = pca.transform(np.array(input_motion.trajectory[i]).reshape(1, -1))[0]
			arriving_pca += [pca_new_point[0]]

			warping.update_warp(pca_new_point)

			matched_pos_in_ref = warping.evaluated_warp[-1]


			draw_ref_colors = [[0.05,0.05,0.05,1] if p > matched_pos_in_ref or p < matched_pos_in_ref-trail_length else [1,0,0,1]\
			                   for p in range(len(ref_motion.trajectory))]

			for plot in self.right_plot_items:
				self.right_plot_view.removeItem(plot)
			self.right_plot_items = []

			for plot in self.left_plot_items:
				self.left_plot_view.removeItem(plot)
			self.left_plot_items = []

			for marker_index in range(0,len(ref_motion.markers)):
				marker_trajectory = [p[3*marker_index:3*marker_index+3] for p in \
						ref_motion.trajectory]

				current_marker_right_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory), color=np.array(draw_ref_colors))
				self.right_plot_view.addItem(current_marker_right_plot)
				self.right_plot_items.append(current_marker_right_plot)

			for marker_index in range(0,len(ref_motion.markers)):
				time_colors = [3 * [float(a) / trail_length] \
						for a in range(0, int(trail_length))]

				marker_trajectory = [p[3*marker_index:3*marker_index+3] for p in \
						input_motion.trajectory[i-trail_length:i]]

				current_marker_left_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory), color=np.array(time_colors))
				self.right_plot_view.addItem(current_marker_left_plot)
				self.right_plot_items.append(current_marker_left_plot)

		self.latest_computed_warp = warping.evaluated_warp
			# if last_matched_pos_in_ref != matched_pos_in_ref:
			# 	plot_warp.clear()
			# 	plot_warp.plot(range(len(warping.evaluated_warp)),warping.evaluated_warp)
			# 	self.wait_for_space()
			# 	last_matched_pos_in_ref = matched_pos_in_ref

		self.latest_computed_warp = warping.match_list

		plot_warp = pg.plot(title="plot warp")
		a = list(zip(*warping.match_list)[0])
		b = list(zip(*warping.match_list)[1])
		self.print_c(str(warping.match_list))
		plot_warp.plot(a,b,symbol='o')
		self.print_c("Done matching.")
		_static_non_blocking_wait(self.print_c, 1)

	def prediction_one_on_one(self):
		ref_indx = self.traj_list_ref.currentRow()
		self.registered_ref = ref_indx
		if ref_indx == -1:
			self.print_c("No Reference Trajectory Selected")
			return

		input_indx = self.traj_list_reg.currentRow()
		if input_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		input_motion = self.motions_db.motions[input_indx]
		ref_motion = self.motions_db.motions[ref_indx]

		###
		# Build PCA Matrix
		###
		pose_matrix = []
		pose_ranges = []
		for m in self.motions_db.motions:
			a = max(len(pose_matrix),0)
			for p in m.trajectory:
				pose_matrix.append(p)
			pose_ranges += [(a,len(pose_matrix))]

		pose_matrix = np.array(pose_matrix)

		mean_pose_trajectories = []

		for i in range(0, len(pose_matrix)):
			mean_x = np.average([pose_matrix[i][j] for j in range(0, len(pose_matrix[i]), 3)])
			mean_y = np.average([pose_matrix[i][j] for j in range(1, len(pose_matrix[i]), 3)])
			mean_z = np.average([pose_matrix[i][j] for j in range(2, len(pose_matrix[i]), 3)])

			mean_pose_trajectories.append([mean_x,mean_y,mean_z])

			for j in range(0, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_x
			for j in range(1, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_y
			for j in range(2, len(pose_matrix[i]), 3):
				pose_matrix[i][j] = pose_matrix[i][j] - mean_z

		n_components = 3

		pca = PCA(n_components=n_components)
		pca.fit(np.array(pose_matrix))

		###
		# Project onto PCA
		###
		ref_pca_trajectory = pca.transform(ref_motion.trajectory)

		###
		# Find Latest Warp and Warp Ref trajectory in PCA
		###
		warping = self.latest_computed_warp

		transpose_warp = np.transpose(warping)

		warping_function = interpolate.interp1d(transpose_warp[0],transpose_warp[1])
		warp = [warping_function(i) for i in range(warping[-1][0])]

		butter_smooth = False

		if butter_smooth:
			self.butter_b, self.butter_a = signal.butter(2, 0.05, 'low', analog=False)
			first_p = warp[0]
			to_filter_list = warp
			for i in range(100):
				to_filter_list.insert(0, first_p)
			warp = signal.lfilter(self.butter_b, self.butter_a, to_filter_list, axis=0)[100:]

		warp_plot = pg.plot(title="warp")
		warp_plot.plot(warp)

		warped_ref_pca_traj = np.transpose(map(lambda x: _static_warp_motion(warp, x, interpolate=True),
		                                       np.transpose(ref_pca_trajectory)))

		input_pca_trajectory = pca.transform(input_motion.trajectory)[:warping[-1][0]]

		reconstructed_warp_traj = pca.inverse_transform(warped_ref_pca_traj)

		ref_motion_range = pose_ranges[ref_indx]
		warped_mean_trajectories = np.transpose(map(lambda x: _static_warp_motion(warp, x, interpolate=True),
		                                       np.transpose(mean_pose_trajectories[ref_motion_range[0]:ref_motion_range[1]])))

		input_reconstructed = pca.inverse_transform(input_pca_trajectory)

		input_motion_start = pose_ranges[input_indx][0]
		for i in range(0, len(reconstructed_warp_traj)):
			warped_ref_centroid = warped_mean_trajectories[i]
			input_centroid = mean_pose_trajectories[input_motion_start+i]
			for j in range(0, len(reconstructed_warp_traj[i]), 3):
				reconstructed_warp_traj[i][j] += warped_ref_centroid[0]
			for j in range(1, len(reconstructed_warp_traj[i]), 3):
				reconstructed_warp_traj[i][j] += warped_ref_centroid[1]
			for j in range(2, len(reconstructed_warp_traj[i]), 3):
				reconstructed_warp_traj[i][j] += warped_ref_centroid[2]

			for j in range(0, len(input_reconstructed[i]), 3):
				input_reconstructed[i][j] += input_centroid[0]
			for j in range(1, len(input_reconstructed[i]), 3):
				input_reconstructed[i][j] += input_centroid[1]
			for j in range(2, len(input_reconstructed[i]), 3):
				input_reconstructed[i][j] += input_centroid[2]

		warped_ref_traj = np.transpose(map(lambda x: _static_warp_motion(warp, x, interpolate=True),
		                                       np.transpose(ref_motion.trajectory)))

		###
		# Compute delayed Diff
		###
		latency = 0.05
		motion_rate = 100
		FRAME_TIME = 1.0/motion_rate
		frames_of_delay = int(latency*motion_rate)

		diff = reconstructed_warp_traj - input_reconstructed

		delayed_diff = frames_of_delay * [np.zeros(len(diff[0])).tolist()] + diff.tolist()

		corrected_warped_traj = warped_ref_traj - np.array(delayed_diff)[:-frames_of_delay]

		###
		# Setup Kalman Filter For Comparison
		###

		mk_indx = -1
		for marker_index in range(len(input_motion.markers)):
			if input_motion.markers[marker_index] == "liu:LFIN":
				mk_indx = marker_index

		i = 3*mk_indx
		target_marker_trajectory = [[a[i],a[i+1],a[i+2]] for a in input_motion.trajectory]


		OBS_VAR, PROC_VAR = 0.001, 0.1
		p_e, v_e, a_e = np.array([0.5 * FRAME_TIME ** 2, FRAME_TIME, 2.0]) * PROC_VAR

		initial_state = np.array(
			[input_motion.trajectory[0][0], input_motion.trajectory[0][1], input_motion.trajectory[0][2], 0, 0, 0, 0, 0,
			 0])
		initial_cov = np.diag([p_e, p_e, p_e, v_e, v_e, v_e, a_e, a_e, a_e])

		transition_matrix = np.array([[1., 0., 0., FRAME_TIME, 0., 0., 0.5 * FRAME_TIME ** 2, 0., 0.],
									  [0., 1., 0., 0., FRAME_TIME, 0., 0., 0.5 * FRAME_TIME ** 2, 0.],
									  [0., 0., 1., 0., 0., FRAME_TIME, 0., 0., 0.5 * FRAME_TIME ** 2],
									  [0., 0., 0., 1., 0., 0., FRAME_TIME, 0., 0.],
									  [0., 0., 0., 0., 1., 0., 0., FRAME_TIME, 0.],
									  [0., 0., 0., 0., 0., 1., 0., 0., FRAME_TIME],
									  [0., 0., 0., 0., 0., 0., 0., 0., 0.],
									  [0., 0., 0., 0., 0., 0., 0., 0., 0.],
									  [0., 0., 0., 0., 0., 0., 0., 0., 0.]])

		observation_cov = np.diag([OBS_VAR, OBS_VAR, OBS_VAR])
		observation_matrix = np.array([[1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2, 0, 0],
									   [0, 1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2, 0],
									   [0, 0, 1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2]])

		state_mean = [initial_state]
		state_cov = [initial_cov]

		kf = KalmanFilter(
			transition_matrices=transition_matrix,
			observation_matrices=observation_matrix,
			initial_state_mean=initial_state,
			initial_state_covariance=initial_cov,
			transition_covariance=initial_cov,
			observation_covariance=observation_cov
		)

		delayed_positions = (frames_of_delay) * [target_marker_trajectory] + target_marker_trajectory

		for i in range(frames_of_delay, warping[-1][0]):
			_static_non_blocking_wait(self.print_c, 0.00001)

			for plot in self.right_plot_items:
				self.right_plot_view.removeItem(plot)
			self.right_plot_items = []

			for marker_index in range(len(ref_motion.markers)):

				warped_ref_marker_trajectory = [p[3*marker_index:3*marker_index+3] for p in \
				                                corrected_warped_traj[i-2:i]]

				if ref_motion.markers[marker_index] == "liu:LFIN":
					current_marker_right_plot = pggl.GLLinePlotItem(pos=np.array(warped_ref_marker_trajectory), \
																	color=np.array([[1, 0, 0, 1] for a in range(
																		len(warped_ref_marker_trajectory))]))
				else:
					current_marker_right_plot = pggl.GLLinePlotItem(pos=np.array(warped_ref_marker_trajectory), \
																	color=np.array([[0, 1, 0, 1] for a in range(
																		len(warped_ref_marker_trajectory))]))

				self.right_plot_view.addItem(current_marker_right_plot)
				self.right_plot_items.append(current_marker_right_plot)

			for marker_index in range(len(input_motion.markers)):

				input_traj_markers = [p[3*marker_index:3*marker_index+3] for p in \
				                                input_motion.trajectory[i-2:i]]

				current_marker_right_plot = pggl.GLLinePlotItem(pos=np.array(input_traj_markers),\
				                                                color=np.array([[1,1,1,1] for a in range(len(input_traj_markers))]))
				self.right_plot_view.addItem(current_marker_right_plot)
				self.right_plot_items.append(current_marker_right_plot)

	def regular_kalman_prediction(self):
		reg_indx = self.traj_list_reg.currentRow()

		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		ref_indx = self.traj_list_reg.currentRow()

		if ref_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		input_motion = self.motions_db.motions[reg_indx]

		trajectory = []

		FRAME_DELAY, FRAME_TIME, OBS_VAR, PROC_VAR = 4, 0.01, 0.000001, 1000
		p_e, v_e, a_e = np.array([0.5 * FRAME_TIME ** 2, FRAME_TIME, 2.0]) * PROC_VAR

		initial_state = np.array([input_motion.trajectory[0][0],input_motion.trajectory[0][1],input_motion.trajectory[0][2], 0,0,0, 0,0,0])
		initial_cov = np.diag([p_e,p_e, p_e, v_e,v_e,v_e, a_e,a_e,a_e])

		transition_matrix = np.array([[1., 0., 0., FRAME_TIME, 0., 0., 0.5*FRAME_TIME**2, 0., 0.],
								       [0., 1., 0., 0., FRAME_TIME, 0., 0., 0.5*FRAME_TIME**2, 0.],
								       [0., 0., 1., 0., 0., FRAME_TIME, 0., 0., 0.5*FRAME_TIME**2],
								       [0., 0., 0., 1., 0., 0., FRAME_TIME, 0., 0.],
								       [0., 0., 0., 0., 1., 0., 0., FRAME_TIME, 0.],
								       [0., 0., 0., 0., 0., 1., 0., 0., FRAME_TIME],
								       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
								       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
								       [0., 0., 0., 0., 0., 0., 0., 0., 0.]])

		observation_cov = np.diag([OBS_VAR,OBS_VAR,OBS_VAR])
		observation_matrix = np.array([[1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2, 0, 0],
		                               [0, 1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2, 0],
		                               [0, 0, 1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2]])

		state_mean = [initial_state]
		state_cov = [initial_cov]

		kf = KalmanFilter(
			transition_matrices=transition_matrix,
			observation_matrices=observation_matrix,
			initial_state_mean=initial_state,
			initial_state_covariance=initial_cov,
			transition_covariance=initial_cov,
			observation_covariance=observation_cov
		)

		delayed_positions = (FRAME_DELAY + 1) * [list(input_motion.trajectory[0])] + list(input_motion.trajectory)

		filtered_plot = pggl.GLLinePlotItem()
		self.right_plot_view.addItem(filtered_plot)
		expected_plot = pggl.GLLinePlotItem()
		self.right_plot_view.addItem(expected_plot)

		for t in range(1, len(delayed_positions)):

			pos_t = np.array(delayed_positions[t])

			new_state, new_cov = kf.filter_update(
				np.array(state_mean[-1]),
				np.array(state_cov[-1]),
				np.array([pos_t[0],pos_t[1],pos_t[2]])
			)

			state_mean.append(new_state)
			state_cov.append(new_cov)

			[px,py,pz,dpdtx,dpdty,dpdtz,d2pdt2x,d2pdt2y,d2pdt2z] = state_mean[-1]

			p_t, dp_dt = np.array([px,py,pz]), np.array([dpdtx,dpdty,dpdtz])
			filtered_obj_pos = p_t + FRAME_DELAY * FRAME_TIME * dp_dt

			trajectory.append(filtered_obj_pos)


			filtered_plot.setData(pos=np.array(trajectory), color=np.array(len(trajectory) * [[1, 0, 0, 1]]))

			expected_plot.setData(pos=np.array(input_motion.trajectory[:t]), color=np.array(t * [[1, 1, 1, 1]]))

			_static_non_blocking_wait(self.print_c, 0.01)

		blur_plot = pg.PlotWidget()
		blur_plot.plot([np.linalg.norm(trajectory[i] - input_motion.trajectory[i])
		                for i in range(min(len(trajectory), len(input_motion.trajectory)))])

		blur_plot.setWindowTitle("Blur Trajectory")
		blur_plot.show()

		_static_non_blocking_wait(self.print_c, 2)
		self.right_plot_view.removeItem(filtered_plot)
		self.right_plot_view.removeItem(expected_plot)

	def kalman_one_on_one(self):
		reg_indx = self.traj_list_reg.currentRow()

		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		ref_indx = self.traj_list_ref.currentRow()

		if ref_indx == -1:
			self.print_c("No Reference Trajectory Selected")
			return

		input_motion = self.motions_db.motions[reg_indx]
		ref_motion = self.motions_db.motions[ref_indx]

		trajectory = []

		FRAME_DELAY, FRAME_TIME, OBS_VAR, PROC_VAR = 4, 0.01, 0.000001, 1000
		p_e, v_e, a_e = np.array([0.5 * FRAME_TIME ** 2, FRAME_TIME, 2.0]) * PROC_VAR

		initial_state = np.array([input_motion.trajectory[0][0],input_motion.trajectory[0][1],input_motion.trajectory[0][2], 0,0,0, 0,0,0])
		initial_cov = np.diag([p_e,p_e, p_e, v_e,v_e,v_e, a_e,a_e,a_e])

		transition_matrix = np.array([[1., 0., 0., FRAME_TIME, 0., 0., 0.5*FRAME_TIME**2, 0., 0.],
								       [0., 1., 0., 0., FRAME_TIME, 0., 0., 0.5*FRAME_TIME**2, 0.],
								       [0., 0., 1., 0., 0., FRAME_TIME, 0., 0., 0.5*FRAME_TIME**2],
								       [0., 0., 0., 1., 0., 0., FRAME_TIME, 0., 0.],
								       [0., 0., 0., 0., 1., 0., 0., FRAME_TIME, 0.],
								       [0., 0., 0., 0., 0., 1., 0., 0., FRAME_TIME],
								       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
								       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
								       [0., 0., 0., 0., 0., 0., 0., 0., 0.]])

		observation_cov = np.diag([OBS_VAR,OBS_VAR,OBS_VAR])
		observation_matrix = np.array([[1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2, 0, 0],
		                               [0, 1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2, 0],
		                               [0, 0, 1, 0, 0, FRAME_TIME, 0, 0, FRAME_TIME ** 2]])

		state_mean = [initial_state]
		state_cov = [initial_cov]

		kf = KalmanFilter(
			transition_matrices=transition_matrix,
			observation_matrices=observation_matrix,
			initial_state_mean=initial_state,
			initial_state_covariance=initial_cov,
			transition_covariance=initial_cov,
			observation_covariance=observation_cov
		)

		delayed_positions = (FRAME_DELAY + 1) * [list(input_motion.trajectory[0])] + list(input_motion.trajectory)

		filtered_plot = pggl.GLLinePlotItem()
		self.right_plot_view.addItem(filtered_plot)
		expected_plot = pggl.GLLinePlotItem()
		self.right_plot_view.addItem(expected_plot)

		vel_etimate_plot = pggl.GLLinePlotItem()
		self.left_plot_view.addItem(vel_etimate_plot)

		warp = OnlineDynamicTimeWarp(ref_motion)

		for t in range(1, len(delayed_positions)):

			pos_t = np.array(delayed_positions[t])

			warp.update_warp(pos_t)

			indx = warp.evaluated_warp[-1]

			diff_t = (ref_motion.trajectory[min(indx+FRAME_DELAY,len(ref_motion.trajectory)-1)] - ref_motion.trajectory[indx]) / float(FRAME_DELAY)

			new_state, new_cov = kf.filter_update(
				np.array(state_mean[-1]),
				np.array(state_cov[-1]),
				np.array([pos_t[0],pos_t[1],pos_t[2]])
			)

			state_mean.append(new_state)
			state_cov.append(new_cov)

			[px,py,pz,dpdtx,dpdty,dpdtz,d2pdt2x,d2pdt2y,d2pdt2z] = state_mean[-1]

			p_t, dp_dt = np.array([px,py,pz]), np.array([dpdtx,dpdty,dpdtz])
			filtered_obj_pos = pos_t + FRAME_DELAY * diff_t

			trajectory.append(filtered_obj_pos)

			filtered_plot.setData(pos=np.array(trajectory), color=np.array(len(trajectory) * [[1, 0, 0, 1]]))

			expected_plot.setData(pos=np.array(input_motion.trajectory[:t]), color=np.array(t * [[1, 1, 1, 1]]))

			vel_etimate_plot.setData(pos=np.array([[0,0,0],10*diff_t]),color=np.array(2*[[1,1,1,1]]))

			_static_non_blocking_wait(self.print_c, 0.1)

		blur_plot = pg.PlotWidget()

		blur_plot.plot([np.linalg.norm(trajectory[i] - input_motion.trajectory[i])
		                for i in range(min(len(trajectory), len(input_motion.trajectory)))])

		blur_plot.setWindowTitle("Blur Trajectory")
		blur_plot.resize(450, 200)
		blur_plot.show()

		_static_non_blocking_wait(self.print_c, 2)
		self.right_plot_view.removeItem(filtered_plot)
		self.right_plot_view.removeItem(expected_plot)

		self.left_plot_view.removeItem(vel_etimate_plot)

	def list_change_event(self, current, previous):
		#self.refresh_plot()
		self.print_c("")

	def refresh_traj_list(self):
		i = 0
		self.traj_list_reg.clear()
		self.traj_list_ref.clear()
		for t in self.motions_db.motions:
			i += 1
			li1 = QtWidgets.QListWidgetItem()
			li1.setText(t.name)
			li2 = QtWidgets.QListWidgetItem()
			li2.setText(t.name)
			self.traj_list_ref.insertItem(i, li1)
			self.traj_list_reg.insertItem(i, li2)

		self.traj_list_ref.setMaximumWidth(200)
		self.traj_list_ref.setMaximumHeight(300)
		self.traj_list_reg.setMaximumWidth(200)
		self.traj_list_reg.setMaximumHeight(300)

	def refresh_all(self):
		self.refresh_traj_list()


class TwistMatchingWidget(QtWidgets.QWidget):
	def __init__(self):

		self.col = -1

		self.is_show_splines = False
		self.is_registering = False
		self.is_prediction_showcase = False

		self.registered_ref = -1

		super(TwistMatchingWidget, self).__init__()

		self.print_c = None

		self.motions_db = Motion3DDatabase()

		self.spacePressed = False
		self.lastTimeSpacePressed = time.time()
		self.latest_computed_warp = []

		self.distance_matrix = []

		self.warp_type = "nothing"

		self.time_slider = QtWidgets.QSlider()
		self.time_slider.setOrientation(QtCore.Qt.Orientation.Vertical)
		self.time_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksAbove)
		self.time_slider.valueChanged.connect(self.slider_moved)

		self.traj_list_ref = QtWidgets.QListWidget()
		self.traj_list_ref.currentItemChanged.connect(self.list_change_event)

		self.traj_list_reg = QtWidgets.QListWidget()
		self.traj_list_reg.currentItemChanged.connect(self.list_change_event)

		self.markers_list = QtWidgets.QListWidget()
		self.markers_list.currentItemChanged.connect(self.markers_list_change_event)
		self.markers_list.setMaximumWidth(200)

		self.simple_twist_button = QtWidgets.QPushButton('Single Twists Warp')
		self.simple_twist_button.clicked.connect(self.simple_twists_button_event)

		self.n_squared_twists_button = QtWidgets.QPushButton('n^2 Twists Warp')
		self.n_squared_twists_button.clicked.connect(self.n_squared_twists_button_event)

		self.simple_markers_button = QtWidgets.QPushButton('Simpler markers Warp')
		self.simple_markers_button.clicked.connect(self.simple_markers_button_event)

		self.n_squared_markers_button = QtWidgets.QPushButton('n^2 markers Warp')
		self.n_squared_markers_button.clicked.connect(self.n_squared_markers_button_event)

		self.left_plot_view = pggl.GLViewWidget()
		self.left_plot_items = []
		self.right_plot_view = pggl.GLViewWidget()
		self.right_plot_items = []

		xgrid = pggl.GLGridItem()
		ygrid = pggl.GLGridItem()
		zgrid = pggl.GLGridItem()
		self.left_plot_view.addItem(xgrid)
		self.left_plot_view.addItem(ygrid)
		self.left_plot_view.addItem(zgrid)

		self.right_plot_view.addItem(xgrid)
		self.right_plot_view.addItem(ygrid)
		self.right_plot_view.addItem(zgrid)

		self.layout = QtWidgets.QGridLayout()

		self.layout.addWidget(QtWidgets.QLabel("Reference Motions: "), 0, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.traj_list_ref, 1, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.simple_twist_button, 2, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.n_squared_twists_button, 3, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.simple_markers_button, 4, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.n_squared_markers_button, 5, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(QtWidgets.QLabel("Register Motion: "), 9, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.traj_list_reg, 10, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(QtWidgets.QLabel("Rigid Parts: "), 11, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.markers_list, 12, 0, QtCore.Qt.AlignmentFlag.AlignTop)
		self.layout.addWidget(self.left_plot_view, 0, 1, -1, 1)
		self.layout.addWidget(self.time_slider, 0, 2, -1, 1)
		self.layout.addWidget(self.right_plot_view, 0, 3, -1, 1)


		verticalSpacer = QtWidgets.QSpacerItem(100, 3000, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
		self.layout.addItem(verticalSpacer, 9, 0, QtCore.Qt.AlignmentFlag.AlignTop)

		self.setLayout(self.layout)

		self.current_warp = None

		self.warp_path = []

		self.prev_ref_pos = 0

	def simple_twists_button_event(self):
		self.warp_type = "simple twists"
		self.refresh_warp_path()
		plot = pg.plot(title="Warp")
		plot.plot(self.warp_path[0], self.warp_path[1])

	def n_squared_twists_button_event(self):
		self.warp_type = "n^2 twists"
		self.distance_matrix = self.refresh_warp_path()
		plot = pg.plot(title="Warp")
		plot.plot(self.warp_path[0], self.warp_path[1])

	def simple_markers_button_event(self):
		self.warp_type = "simple markers"
		self.refresh_warp_path()
		plot = pg.plot(title="Warp")
		plot.plot(self.warp_path[0], self.warp_path[1])

	def n_squared_markers_button_event(self):
		self.warp_type = "n^2 markers"
		self.refresh_warp_path()
		plot = pg.plot(title="Warp")
		plot.plot(self.warp_path[0], self.warp_path[1])

	def keyPressEvent(self, event):
		#super(MotionEditionWidget, self).keyPressEvent(event)
		self.on_space_key()

	def on_space_key(self):
		if time.time() - self.lastTimeSpacePressed > 0.2:
			self.spacePressed = True
			self.lastTimeSpacePressed = time.time()

	def wait_for_space(self):
		while not self.spacePressed:
			self.print_c("")
		self.spacePressed = False

	def detect_space(self):
		if self.spacePressed:
			self.spacePressed = False
			return True
		return False

	def list_change_event(self, current, previous):
		index = self.traj_list_reg.currentRow()

		if index > -1:
			motion = self.motions_db.motions[index]
			if self.time_slider.value() >= len(motion.trajectory):
				self.time_slider.setValue(len(motion.trajectory) - 1)
			self.time_slider.setMaximum(len(motion.trajectory))

		self.refresh_warp_path()
		self.refresh_markers_list()
		self.refresh_plots()

	def refresh_traj_list(self):
		i = 0
		self.traj_list_reg.clear()
		self.traj_list_ref.clear()
		for t in self.motions_db.motions:
			i += 1
			li1 = QtWidgets.QListWidgetItem()
			li1.setText(t.name)
			li2 = QtWidgets.QListWidgetItem()
			li2.setText(t.name)
			self.traj_list_ref.insertItem(i, li1)
			self.traj_list_reg.insertItem(i, li2)

		self.traj_list_ref.setMaximumWidth(200)
		self.traj_list_ref.setMaximumHeight(300)
		self.traj_list_reg.setMaximumWidth(200)
		self.traj_list_reg.setMaximumHeight(300)

	def markers_list_change_event(self):
		self.print_c("")

	def refresh_markers_list(self):
		index = self.traj_list_reg.currentRow()
		if index > -1:
			motion = self.motions_db.motions[index]
			i = 0
			self.markers_list.clear()
			for m_name in motion.markers:
				i += 1
				li = QtWidgets.QListWidgetItem()
				li.setText(m_name)
				self.markers_list.insertItem(i, li)

		self.markers_list.setMaximumWidth(200)

	def refresh_all(self):
		self.refresh_traj_list()
		self.refresh_plots()

	def refresh_warp_path(self):
		ref_indx = self.traj_list_ref.currentRow()
		self.registered_ref = ref_indx
		if ref_indx == -1:
			self.print_c("No Reference Trajectory Selected")
			return np.array([])

		reg_indx = self.traj_list_reg.currentRow()
		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return np.array([])

		input_motion = self.motions_db.motions[reg_indx]
		ref_motion = self.motions_db.motions[ref_indx]

		if self.warp_type == "n^2 twists":
			input_trajectory = []
			for rigid_part in input_motion.n_squared_twists_trajectories:
				part_traj = np.transpose(input_motion.n_squared_twists_trajectories[rigid_part])

				for dim in part_traj:
					input_trajectory.append(dim)

			ref_trajectory = []
			for rigid_part in ref_motion.n_squared_twists_trajectories:
				part_traj = np.transpose(ref_motion.n_squared_twists_trajectories[rigid_part])

				for dim in part_traj:
					ref_trajectory.append(dim)

			input_trajectory = np.transpose(input_trajectory)
			ref_trajectory = np.transpose(ref_trajectory)

		elif self.warp_type == "simple twists":
			input_trajectory = []
			input_twists = twist_from_rigid_trajectories(input_motion)
			for rigid_part in input_twists:
				part_traj = np.transpose(input_twists[rigid_part])

				for dim in part_traj:
					input_trajectory.append(dim)

			ref_trajectory = []
			ref_twists = twist_from_rigid_trajectories(ref_motion)
			for rigid_part in ref_twists:
				part_traj = np.transpose(ref_twists[rigid_part])

				for dim in part_traj:
					ref_trajectory.append(dim)

			input_trajectory = np.transpose(input_trajectory)
			ref_trajectory = np.transpose(ref_trajectory)

		elif self.warp_type == "simple markers":
			input_trajectory = input_motion.trajectory
			ref_trajectory = ref_motion.trajectory

		elif self.warp_type == "n^2 markers":
			input_trajectory = []
			for i in range(0, len(input_motion.trajectory)):
				dense_frame = []
				for j in range(int(len(input_motion.trajectory[i]) / 3)):
					for h in range(int(len(input_motion.trajectory[i]) / 3)):
						if j == h:
							continue
						dense_frame += (
						np.array(input_motion.trajectory[i][3 * j:3 * j + 3]) - np.array(input_motion.trajectory[i][3 * h:3 * h + 3])).tolist()
				input_trajectory += [dense_frame]

			ref_trajectory = []
			for i in range(0, len(ref_motion.trajectory)):
				dense_frame = []
				for j in range(int(len(ref_motion.trajectory[i]) / 3)):
					for h in range(int(len(ref_motion.trajectory[i]) / 3)):
						if j == h:
							continue
						dense_frame += (
							np.array(ref_motion.trajectory[i][3 * j:3 * j + 3]) - np.array(
								ref_motion.trajectory[i][3 * h:3 * h + 3])).tolist()
				ref_trajectory += [dense_frame]
			ref_trajectory = np.array(input_trajectory)
			ref_trajectory = np.array(ref_trajectory)
		else:
			return np.array([])

		self.print_c("Computing Warp...")
		import dtw
		dist, cost, acc, self.warp_path = dtw.dtw(input_trajectory, ref_trajectory, lambda x, y: np.linalg.norm(x - y))
		self.print_c("...Done")

		return_dist_matrix = False

		if return_dist_matrix == True:
			dist_matrix = np.zeros((len(input_trajectory), len(ref_trajectory)))
			for i in range(len(input_trajectory)):
				for j in range(len(ref_trajectory)):
					dist_matrix[i,j] = np.linalg.norm(input_trajectory[i] - ref_trajectory[j])
			return dist_matrix
		else:
			return np.array([])

	def slider_moved(self):
		self.refresh_plots()

	def refresh_plots(self):
		ref_indx = self.traj_list_ref.currentRow()
		self.registered_ref = ref_indx
		if ref_indx == -1:
			self.print_c("No Reference Trajectory Selected")
			return

		reg_indx = self.traj_list_reg.currentRow()
		if reg_indx == -1:
			self.print_c("No Registration Trajectory Selected")
			return

		input_motion = self.motions_db.motions[reg_indx]
		ref_motion = self.motions_db.motions[ref_indx]

		_clear_plot(self.left_plot_view, self.left_plot_items)
		_clear_plot(self.right_plot_view, self.right_plot_items)

		time_pos = self.time_slider.value()

		ref_time_pos = 0
		if len(self.distance_matrix) < 1:
			for reg_index in self.warp_path[0]:
				if reg_index == time_pos:
					ref_time_pos = self.warp_path[1][reg_index]
					break
		else:
			ref_time_pos = np.argmin(self.distance_matrix[time_pos][max(0,self.prev_ref_pos -10):10+self.prev_ref_pos])

		if len(input_motion.bones) < 1 or len(ref_motion.bones) < 1:
			_draw_markers_to_plot(self.left_plot_view, self.left_plot_items, input_motion, time_pos, 30, colors=[1, 1, 1, 1])
			_draw_markers_to_plot(self.left_plot_view, self.left_plot_items, ref_motion, ref_time_pos, 30, colors=[1, 0, 0, 1])

			_draw_markers_to_plot(self.right_plot_view, self.right_plot_items, input_motion, time_pos, 30, colors=[1, 1, 1, 1])
			_draw_markers_to_plot(self.right_plot_view, self.right_plot_items, ref_motion, ref_time_pos, 30, colors=[1, 0, 0, 1])
		else:
			_draw_skeleton_to_plot(self.left_plot_view, self.left_plot_items, input_motion, time_pos, colors=[1, 1, 1, 1])
			_draw_skeleton_to_plot(self.left_plot_view, self.left_plot_items, ref_motion, ref_time_pos, colors=[1, 0, 0, 1])

			_draw_skeleton_to_plot(self.right_plot_view, self.right_plot_items, input_motion, time_pos, colors=[1, 1, 1, 1])
			_draw_skeleton_to_plot(self.right_plot_view, self.right_plot_items, ref_motion, time_pos, colors=[1, 0, 0, 1])

		self.prev_ref_pos = ref_time_pos

class GuiApp(object):
	def __init__(self):

		self.columns = [False, False, False, False]

		self.print_c = self.print_to_console

		self.motions_db = Motion3DDatabase()

		self.motions_db.print_c = self.print_to_console

		self.motion_edit_widget = None
		self.natnet_widget = None
		self.traj_prediction_widget = None
		self.compute_eigen_widget = None
		self.filter_traj_widget = None
		self.online_filter_traj_widget = None
		self.traj_analysis_widget = None
		self.twist_widget = None

		# Always start by initializing Qt (only once per application)
		self.app = QtWidgets.QApplication([])
		# Define a top-level widget to hold everything
		self.window = QtWidgets.QMainWindow()
		self.app.setActiveWindow(self.window)
		self.window.resize(1280, 720)

		self.window.setWindowTitle("Motion Modeling Editor")

		self.widget = QtWidgets.QWidget()

		# Create main menu
		self.setup_menu()

		self.consoleText = QtWidgets.QTextEdit()
		self.consoleText.setText("Console Output Below:\n")
		self.consoleText.setReadOnly(True)
		self.consoleText.setMaximumHeight(200)

		self.layout = QtWidgets.QGridLayout()
		self.widget.setLayout(self.layout)

		self.layout.addWidget(self.consoleText, 1, 0, 1, -1, QtCore.Qt.AlignmentFlag.AlignBottom)

		self.window.setCentralWidget(self.widget)
		self.window.show()

		timer = QtCore.QTimer(self.app)
		timer.timeout.connect(self.update_all)
		timer.start(10)

		self.import_motion_db()

	def setup_menu(self):
		mainMenu = self.window.menuBar()
		mainMenu.setNativeMenuBar(False)
		fileMenu = mainMenu.addMenu('File')
		viewMenu = mainMenu.addMenu('View')
		editMenu = mainMenu.addMenu('Edit')

		# Add new button
		newFolderButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'New Mocap', self.window)
		newFolderButton.setShortcut('CTRL+SHIFT+N')
		newFolderButton.setStatusTip('New Mocap data')
		newFolderButton.triggered.connect(self.new_motions)
		fileMenu.addAction(newFolderButton)

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

		analysisButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Trajectory Analysis', self.window)
		analysisButton.setShortcut('CTRL+T')
		analysisButton.setStatusTip('toggle Trajectory Analysis')
		analysisButton.triggered.connect(self.toggle_trajectory_analysis_widget)
		viewMenu.addAction(analysisButton)

		predictionButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Trajectory Prediction', self.window)
		predictionButton.setShortcut('CTRL+Y')
		predictionButton.setStatusTip('toggle Trajectory Prediction')
		predictionButton.triggered.connect(self.trajectory_prediction_widget)
		viewMenu.addAction(predictionButton)

		nnAcquButton = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Twist Matching', self.window)
		nnAcquButton.setShortcut('CTRL+N')
		nnAcquButton.setStatusTip('Toggle Twist Matchhing')
		nnAcquButton.triggered.connect(self.toggle_twist_widget)
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
			self.layout.addWidget(self.motion_edit_widget, 0, col, 1, 1, QtCore.Qt.AlignmentFlag.AlignTop)

		else:
			self.columns[self.motion_edit_widget.col] = False
			self.layout.removeWidget(self.motion_edit_widget)
			self.motion_edit_widget.deleteLater()
			self.motion_edit_widget = None
			self.addKFButMenu.setDisabled(True)
			self.delButMenu.setDisabled(True)

	def trajectory_prediction_widget(self):
		if self.traj_prediction_widget == None:
			self.traj_prediction_widget = TrajectoryPredictionWidget()  # LearningTrajectoryWidgetExperiment()
			self.traj_prediction_widget.motions_db = self.motions_db
			self.traj_prediction_widget.print_c = self.print_c
			self.traj_prediction_widget.refresh_all()
			col = self.assign_col()
			self.traj_prediction_widget.col = col
			self.layout.addWidget(self.traj_prediction_widget, 0, col, 1, 1)
		else:
			self.columns[self.traj_prediction_widget.col] = False
			self.layout.removeWidget(self.traj_prediction_widget)
			self.traj_prediction_widget.deleteLater()
			self.traj_prediction_widget = None

	def toggle_trajectory_analysis_widget(self):
		if self.traj_analysis_widget == None:
			self.traj_analysis_widget = TrajectoryAnalysisWidget()
			self.traj_analysis_widget.motions_db = self.motions_db
			self.traj_analysis_widget.print_c = self.print_c
			self.traj_analysis_widget.refresh_all()
			col = self.assign_col()
			self.traj_analysis_widget.col = col
			self.layout.addWidget(self.traj_analysis_widget, 0, col, 1, 1)
		else:
			self.columns[self.traj_analysis_widget.col] = False
			self.layout.removeWidget(self.traj_analysis_widget)
			self.traj_analysis_widget.deleteLater()
			self.traj_analysis_widget = None

	def toggle_twist_widget(self):
		if self.twist_widget == None:
			self.twist_widget = TwistMatchingWidget()
			self.twist_widget.motions_db = self.motions_db
			self.twist_widget.print_c = self.print_c
			self.twist_widget.refresh_all()
			col = self.assign_col()
			self.twist_widget.col = col
			self.layout.addWidget(self.twist_widget, 0, col, 1, 1)
		else:
			self.columns[self.twist_widget.col] = False
			self.layout.removeWidget(self.twist_widget)
			self.twist_widget.deleteLater()
			self.twist_widget = None

	def update_all(self):
		if self.natnet_widget != None:
			self.natnet_widget.update_plot()
			if self.natnet_widget.shouldImport:
				date = time.strftime('[%d-%m-%y] [%Hh-%Mm-%Ss]')
				name = "{}".format(date)
				self.print_c("Importing \"" + name + "\" from NatNet")
				self.motions_db.add_motion(name, self.natnet_widget.acquisition.obj_positions)
				self.natnet_widget.acquisition.obj_positions = []
				self.natnet_widget.acquisition.cam_positions = []
				self.natnet_widget.object_traj_plot.setData(pos=np.array([0,0,0]))
				self.natnet_widget.shouldImport = False

				if self.motion_edit_widget != None:
					self.motion_edit_widget.refresh_all()
			if self.natnet_widget.shouldExit:
				self.columns[self.natnet_widget.col] = False
				self.layout.removeWidget(self.natnet_widget)
				self.natnet_widget.deleteLater()
				# self.natnet_widget.__del__()
				self.natnet_widget = None

		if self.compute_eigen_widget != None:
			self.compute_eigen_widget.refresh_button()

	def load_motions(self):
		file_selection = QtWidgets.QFileDialog()
		file_selection.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
		file_selection.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
		file_selection.setNameFilters(["Mocap C3D (*" + ".c3d" + ")","Mocap C3D ASCII (*" + ".c3dtxt" + ")","Mocap BVH (*" + ".bvh" + ")"])
		file_selection.show()

		if file_selection.exec_():
			filenames = file_selection.selectedFiles()

			for f in filenames:
				path = "/".join(f.split("/")[:-1])
				if "txt" == f.split(".")[-1]:
					self.motions_db.load_c3dtxt(path, f.split("/")[-1])
				elif 'c3d' in f.split(".")[-1]:
					self.motions_db.load_c3d(path, f.split("/")[-1])
				elif 'bvh' in f.split(".")[-1]:
					self.motions_db.load_bvh(path, f.split("/")[-1])
				else:
					print("Could not recognize format")

			self.refresh_all_widgets()

	def new_motions(self):
		self.motions_db = Motion3DDatabase()
		self.motions_db.print_c = self.print_c
		self.refresh_all_widgets()

	def save_motions_db_as(self):
		file_selection = QtWidgets.QFileDialog()
		file_selection.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
		file_selection.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
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

		with open(self.motions_db.dbFilename, 'wb') as fp:
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

			with open(self.motions_db.dbFilename, 'wb') as fp:
				pickle.dump(self.motions_db, fp)

			self.motions_db.print_c = call_back_temp
			self.motions_db.motion_model = motion_model_temp

	def import_motion_db(self):
		filename = ""
		file_selection = QtWidgets.QFileDialog()
		file_selection.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
		file_selection.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
		file_selection.setNameFilters(["Motion Database (*" + DB_FILE_EXTENSION + ")"])
		file_selection.show()

		if file_selection.exec_():
			filename = file_selection.selectedFiles()[0]

		if filename == "":
			return

		self.print_c("Loading DB: " + filename)
		with open(filename, 'rb') as fp:
			self.motions_db = pickle.load(fp)
			self.motions_db.dbFilename = filename
			#Convert 1 marker only databases to new format
			if not hasattr(self.motions_db.motions[0], 'markers'):
				new_motions = []
				for m in self.motions_db.motions:
					motion = Motion3D(m.name)
					motion.markers = ["Marker 1"]
					motion.original_trajectory = m.original_trajectory
					motion.trajectory = motion.original_trajectory
					new_motions.append(motion)
				self.motions_db.motions = new_motions

		self.motions_db.print_c = self.print_c

		if self.motion_edit_widget != None:
			self.motion_edit_widget.motions_db = self.motions_db
			self.motion_edit_widget.refresh_all()
		if self.traj_prediction_widget != None:
			self.traj_prediction_widget.motions_db = self.motions_db
			self.traj_prediction_widget.refresh_all()

	def print_to_console(self, string):
		if len(string) != 0:
			self.consoleText.setText(self.consoleText.toPlainText() + string + "\n")
			scrollbar = self.consoleText.verticalScrollBar()
			scrollbar.setValue(scrollbar.maximum())
		self.app.processEvents()

	def refresh_all_widgets(self):
		if self.motion_edit_widget != None:
			self.motion_edit_widget.refresh_all()
		if self.traj_prediction_widget != None:
			self.traj_prediction_widget.refresh_all()
		if self.traj_analysis_widget != None:
			self.traj_analysis_widget.refresh_all()

	def assign_col(self):
		for i in range(0, len(self.columns)):
			if self.columns[i] == False:
				self.columns[i] = True
				return i

def main():
	gui = GuiApp()
	gui.app.exec_()

if __name__ == "__main__":
	main()


	# def prediction_one_on_one(self):
	# 	ref_indx = self.traj_list_ref.currentRow()
	# 	self.registered_ref = ref_indx
	# 	if ref_indx == -1:
	# 		self.print_c("No Reference Trajectory Selected")
	# 		return
	#
	# 	reg_indx = self.traj_list_reg.currentRow()
	# 	if reg_indx == -1:
	# 		self.print_c("No Registration Trajectory Selected")
	# 		return
	#
	# 	input_motion = self.motions_db.motions[reg_indx]
	# 	ref_motion = self.motions_db.motions[ref_indx]
	#
	# 	pose_matrix = []
	#
	# 	for m in self.motions_db.motions:
	# 		for p in m.trajectory:
	# 			pose_matrix.append(p)
	#
	# 	pose_matrix = np.array(pose_matrix)
	#
	# 	mean_pose_trajectories = []
	#
	# 	for i in range(0, len(pose_matrix)):
	# 		mean_x = np.average([pose_matrix[i][j] for j in range(0, len(pose_matrix[i]), 3)])
	# 		mean_y = np.average([pose_matrix[i][j] for j in range(1, len(pose_matrix[i]), 3)])
	# 		mean_z = np.average([pose_matrix[i][j] for j in range(2, len(pose_matrix[i]), 3)])
	#
	# 		mean_pose_trajectories.append([mean_x,mean_y,mean_z])
	#
	# 		for j in range(0, len(pose_matrix[i]), 3):
	# 			pose_matrix[i][j] = pose_matrix[i][j] - mean_x
	# 		for j in range(1, len(pose_matrix[i]), 3):
	# 			pose_matrix[i][j] = pose_matrix[i][j] - mean_y
	# 		for j in range(2, len(pose_matrix[i]), 3):
	# 			pose_matrix[i][j] = pose_matrix[i][j] - mean_z
	#
	# 	n_components = 3
	#
	# 	pca = PCA(n_components=n_components)
	# 	pca.fit(np.array(pose_matrix))
	#
	# 	ref_pca_trajectory = pca.transform(ref_motion.trajectory)
	#
	# 	input_pca_trajectory = pca.transform(input_motion.trajectory)[:201]
	# 	# 3 components
	# 	# warping =[[0, 0], [37, 53], [48, 62], [54, 67], [70, 84], [109, 123], [123, 136],
	# 	#           [129, 142], [138, 153], [167, 183], [180, 193], [185, 198], [201, 214]]
	#
	# 	# 4 components
	# 	warping = [[0, 0], [37, 53], [48, 62], [54, 67], [59, 70], [70, 84], [109, 123],[121, 133], [123, 136],
	# 	           [129, 142],[138, 153], [167, 183], [180, 193], [185, 198], [189, 203], [201, 214]]
	#
	# 	transpose_warp = np.transpose(warping)
	#
	# 	warping_function = interpolate.interp1d(transpose_warp[0],transpose_warp[1])
	# 	warp = [warping_function(i) for i in range(201)]
	# 	warped_ref_pca_traj = np.transpose(map(lambda x: _static_warp_motion(warp, x, interpolate=True),
	# 	                                       np.transpose(ref_pca_trajectory)))
	#
	# 	displacement = np.array(warped_ref_pca_traj) - np.array(input_pca_trajectory)
	#
	# 	# displacement_plot = pg.plot(title="Warped_PCA + Displacement")
	# 	# for j in range(len(warped_ref_pca_traj[0])):
	# 	# 	warp_ref_pca = [x[j] for x in warped_ref_pca_traj]
	# 	# 	displacement_plot.plot(warp_ref_pca, pen=(j, len(warped_ref_pca_traj[0])))
	# 	# for j in range(len(warped_ref_pca_traj[0])):
	# 	# 	input_pca_traj = [x[j] for x in input_pca_trajectory]
	# 	# 	displacement_plot.plot(10 + np.array(input_pca_traj), pen=(j, len(warped_ref_pca_traj[0])))
	# 	# for j in range(len(warped_ref_pca_traj[0])):
	# 	# 	disp_j = [x[j] for x in displacement]
	# 	# 	displacement_plot.plot(20 + np.array(disp_j), pen=(j, len(warped_ref_pca_traj[0])))
	# 	#
	# 	# for w in warping:
	# 	# 	displacement_plot.addLine(w[0])
	# 	#
	# 	#
	# 	# butter_b, butter_a = signal.butter(2, 0.1, 'low', analog=False)
	# 	#
	# 	# filtered_ref_pca = signal.lfilter(butter_b, butter_a, ref_pca_trajectory, axis=0)
	# 	# filtered_input_pca = signal.lfilter(butter_b, butter_a, input_pca_trajectory, axis=0)
	# 	#
	# 	# filtered_warped_ref_pca = np.transpose(map(lambda x: _static_warp_motion(warp, x, interpolate=True),
	# 	#                                        np.transpose(filtered_ref_pca)))
	# 	#
	# 	# filtered_displacement = np.array(filtered_warped_ref_pca) - np.array(filtered_input_pca)
	# 	#
	# 	# filtered_displacement_plot = pg.plot(title="Filtered Warped_PCA + Displacement")
	# 	# for j in range(len(warped_ref_pca_traj[0])):
	# 	# 	warp_ref_pca = [x[j] for x in filtered_warped_ref_pca]
	# 	# 	filtered_displacement_plot.plot(warp_ref_pca, pen=(j, len(warped_ref_pca_traj[0])))
	# 	# for j in range(len(warped_ref_pca_traj[0])):
	# 	# 	input_pca_traj = [x[j] for x in filtered_input_pca]
	# 	# 	filtered_displacement_plot.plot(10 + np.array(input_pca_traj), pen=(j, len(warped_ref_pca_traj[0])))
	# 	# for j in range(len(warped_ref_pca_traj[0])):
	# 	# 	disp_j = [x[j] for x in filtered_displacement]
	# 	# 	filtered_displacement_plot.plot(20 + np.array(disp_j), pen=(j, len(warped_ref_pca_traj[0])))
	# 	#
	# 	# for w in warping:
	# 	# 	filtered_displacement_plot.addLine(w[0])
	#
	# 	reconstructed_warp_traj = pca.inverse_transform(input_pca_trajectory)
	#
	# 	warped_mean_trajectories = np.transpose(map(lambda x: _static_warp_motion(warp, x, interpolate=False),
	# 	                                       np.transpose(mean_pose_trajectories)))
	#
	# 	for i in range(0, len(reconstructed_warp_traj)):
	# 		(mean_x, mean_y, mean_z) = warped_mean_trajectories[i]
	#
	# 		for j in range(0, len(reconstructed_warp_traj[i]), 3):
	# 			reconstructed_warp_traj[i][j] += mean_x
	# 		for j in range(1, len(reconstructed_warp_traj[i]), 3):
	# 			reconstructed_warp_traj[i][j] += mean_y
	# 		for j in range(2, len(reconstructed_warp_traj[i]), 3):
	# 			reconstructed_warp_traj[i][j] += mean_z
	#
	# 	warped_ref_traj = np.transpose(map(lambda x: _static_warp_motion(warp, x, interpolate=True),
	# 	                                       np.transpose(ref_motion.trajectory)))
	#
	#
	# 	trail_length = 10
	# 	for i in range(trail_length,len(warped_ref_traj)):
	# 		_common_help_non_blocking_wait(self.print_c, 0.00001)
	#
	# 		for plot in self.right_plot_items:
	# 			self.right_plot_view.removeItem(plot)
	# 		self.right_plot_items = []
	#
	# 		for marker_index in range(0, len(ref_motion.markers)):
	# 			ref_colors = [[float(a)/ trail_length,0,0] for a in range(0, int(trail_length))]
	#
	# 			marker_trajectory = [p[3 * marker_index:3 * marker_index + 3] for p in \
	# 			                     reconstructed_warp_traj[i - trail_length:i]]
	#
	# 			current_marker_ref_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory),
	# 			                                               color=np.array(ref_colors))
	#
	# 			marker_trajectory_input = [p[3 * marker_index:3 * marker_index + 3] for p in \
	# 			                     input_motion.trajectory[i - trail_length:i]]
	#
	# 			input_colors = [[0, float(a) / trail_length, 0] for a in range(0, int(trail_length))]
	#
	# 			current_marker_input_plot = pggl.GLLinePlotItem(pos=np.array(marker_trajectory_input),
	# 			                                               color=np.array(input_colors))
	# 			self.right_plot_view.addItem(current_marker_input_plot)
	# 			self.right_plot_view.addItem(current_marker_ref_plot)
	# 			self.right_plot_items.append(current_marker_ref_plot)
	# 			self.right_plot_items.append(current_marker_input_plot)