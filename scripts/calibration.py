#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   calibration.py
@Time    :   2024-07-08 23:13:13
@Author  :   feng li
@Contact :   feng.li@tum.de
@Description    :   
'''

# ros
import rospy
from iiwa_msgs.msg import CartesianPose
from geometry_msgs.msg import PoseStamped

# qt
from PySide6.QtCore import QCoreApplication, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtUiTools import QUiLoader

# modules
from utils import CaliSampler, CaliCompute, Transform, RangeSampler, LoopX

# others
import numpy as np
import os
import sys
import time

class Calibration():
    def __init__(self,
                 project_path,
                 ui_path = '/ui/calibration.ui',
                 sample_path = '/config/cali_samples.txt',
                 result_path = '/config/cali_results.txt',
                 loopx_diff_path = '/config/loopx_diff.txt',
                 range_sample_path = '/config/range_samples.txt',
                 robot_topic = '/iiwa/state/CartesianPose',
                 pub_topic = '/iiwa/command/CartesianPoseLin',
                 cali_method = 'Horaud'):

        rospy.init_node('calibration_node', anonymous=True)
        self.sample_path = project_path + sample_path
        self.result_path = project_path + result_path
        self.cali_method = cali_method
        self.command_pub = rospy.Publisher(pub_topic, PoseStamped, queue_size=1)

        # UI and Button
        self.ui = QUiLoader().load(project_path + ui_path)
        ## manual range marking
        self.ui.range_sample.clicked.connect(self.handle_range_sample)
        self.ui.range_measure.clicked.connect(self.handle_range_measure)
        ## robot control
        self.ui.move_rob.clicked.connect(self.handle_move)
        self.ui.stop_rob.clicked.connect(self.handle_stop)
        ## calibration
        self.ui.pose_compute.clicked.connect(self.handle_pose_compute)
        self.ui.sample.clicked.connect(self.handle_sample)
        self.ui.auto_sample.clicked.connect(self.handle_auto_sample)
        self.ui.compute.clicked.connect(self.handle_compute)
        self.ui.reset.clicked.connect(self.handle_reset)
        self.ui.quit.clicked.connect(QApplication.instance().quit)

        # Operated module
        self.sample = CaliSampler(self.sample_path, self.result_path)
        self.range_sample = RangeSampler(project_path + range_sample_path)
        self.compute = CaliCompute()
        self.loopx = LoopX()
        self.tf = Transform()

        self.loopx_diff = self.compute.get_coordinate(project_path + loopx_diff_path)

        # initialization
        self.sample_cnt = 0
        self.range_sample_cnt = 0
        self.sample.save_sample_init()
        ## You can not initialize it if the position of the iiwa and loop x don't change.
        # self.range_sample.save_sample_init()

        # Robot iiwa
        self.robot_pose = None
        self.robot_pose_7d = []
        self.desired_pose = None
        self.robot_stopped = True
        self.max_motion_time = 60
        try:
            rospy.Subscriber(robot_topic, CartesianPose, self.rob_callback)
            rospy.wait_for_message(robot_topic, CartesianPose, timeout=1)
        except rospy.exceptions.ROSException as e:
            print("No robot topic. Please check it.")

        # Loop X
        self.loop_x_pose = []
        self.cam_t = []
        self.cam_q = []
        rospy.Timer(rospy.Duration(0.1), self.get_loop_x_pose)

        # Timers
        rospy.Timer(rospy.Duration(0.01), self.show_robot_states)
        rospy.Timer(rospy.Duration(0.05), self.robot_motion_detection)

        self.auto_sample_timer = QTimer()
    
    def rob_callback(self, ros_msg):
        self.robot_pose = ros_msg.poseStamped.pose
        self.robot_state_header = ros_msg.poseStamped.header

    def get_loop_x_pose(self, event):
        self.loop_x_pose, notes = self.loopx.get_pose_loopx()
        # print(notes)
        self.cam_t = self.loop_x_pose[:3]
        self.cam_q = self.loop_x_pose[3:]

    def show_robot_states(self, event):
        if self.robot_pose is not None:
            tvec = self.robot_pose.position
            qvec = self.robot_pose.orientation
            self.rob_t = [tvec.x, tvec.y, tvec.z]
            self.rob_q = [qvec.x, qvec.y, qvec.z, qvec.w]
            self.robot_pose_7d = np.array(self.rob_t + self.rob_q)
            
            ## Show robot end-effector pose
            rob_cur_pose_show = self.tf.quat_to_angle(self.robot_pose_7d, unit_m_to_mm=True)
            rob_cur_pose_show = np.round(rob_cur_pose_show, decimals=3)
            self.ui.robot_pose.setText(self.tf.fstr(rob_cur_pose_show))
        else:
            self.ui.robot_pose.setText('Do not get pose!')

        if len(self.loop_x_pose) != 0:
            try:
                loop_x_cur_pose_show = self.tf.quat_to_angle(self.loop_x_pose, unit_m_to_mm=True)
                loop_x_cur_pose_show = np.round(loop_x_cur_pose_show, decimals=3)
                self.ui.loop_x_pose.setText(self.tf.fstr(loop_x_cur_pose_show))
            except Exception as e:
                print("Loopx pose show problem.")
        else:
            self.ui.loop_x_pose.setText('Do not get pose!')

    def handle_range_sample(self):
        if self.robot_pose is None:
            return
        self.range_sample_cnt += 1
        self.ui.sample_range_count.setText('Sample times: ' + str(self.range_sample_cnt))
        self.range_sample.save_sample(self.rob_t, self.rob_q)

        self.show_range_sample()

    def handle_range_measure(self):
        try:
            ee_to_base = self.range_sample.get_sample()
        except Exception as e:
            print("Didn't get range!")
            QMessageBox.about(self.ui, 'Warning', 'Did not get range! ')
            return

        ee_to_base_angle_mm = self.tf.pose_7d_to_pose_6d(np.array(ee_to_base), True)
        max_value, min_value = self.tf.get_max_min(ee_to_base_angle_mm)

        table_model = QStandardItemModel()
        table_model.setHorizontalHeaderLabels(['tx(mm)', 'ty(mm)', 'tz(mm)', 'rx(deg)', 'ry(deg)', 'rz(deg)'])
        table_model.appendRow([QStandardItem(str(entry)) for entry in max_value])
        table_model.appendRow([QStandardItem(str(entry)) for entry in min_value])

        self.ui.range_view.setModel(table_model)

    def handle_pose_compute(self):
        t_cam, t_rob = self.sample.read_and_compute_pose()
        cali_res = self.compute.get_calibration_res(self.result_path)
        print(f't_cam: {t_cam}', '\n')
        print(f't_rob: {t_rob}', '\n')
        cali_new = np.dot(cali_res, np.linalg.inv(t_cam))
        print(cali_new)

        rob_pose_path = os.path.join('/home/leofer/ros_projects/loop_x/src/loop_x/config', 'cali_move5.txt')
        self.sample.save_transformation(cali_new, rob_pose_path)

    def handle_sample(self):
        if self.robot_pose is None:
            print("Did not get the robot pose")
            return
        self.sample.save_sample(self.rob_t, self.rob_q, self.cam_t, self.cam_q)
        # self.sample.save_sample_pose(self.rob_t, self.rob_q, self.cam_t, self.cam_q, self.sample_cnt)
        self.sample_cnt += 1
        self.ui.sample_count.setText('Sample times: ' + str(self.sample_cnt))

        self.show_sample()

    def handle_auto_sample(self):
        if not self.robot_stopped:
            QMessageBox.about(self.ui, 'Warning', 'Please wait to finish the movement! ')
            return
        if self.robot_pose is None:
            print("Did not get the robot pose")
            return
        
        # initialization
        self.handle_reset(is_auto=True)
        
        sample_nums = self.ui.sample_nums.text()
        if sample_nums == '':
            QMessageBox.about(self.ui, 'Warning', 'Please input sample number! ')
            return 
        
        self.sample_nums = int(sample_nums)
        if self.sample_nums < 6:
            QMessageBox.about(self.ui, 'Warning', 'Please make sure the sample number is more than 6! ')
            return 
        
        ee_to_base = self.range_sample.get_sample()
        ee_to_base_angle_m = self.tf.pose_7d_to_pose_6d(np.array(ee_to_base), False)
        self.max_value, self.min_value = self.tf.get_max_min(ee_to_base_angle_m)
        if len(self.max_value) == 0 or len(self.min_value) == 0:
            QMessageBox.about(self.ui, 'Warning', 'Did not get max value and min value! ')
            return 
        self.auto_sample_timer.timeout.connect(self.auto_sample_single_task)
        self.auto_sample_timer.start(1000)
        
    def handle_stop(self):
        self.range_sample_cnt = 0
        self.sample_nums = 0
        self.ui.sample_nums.setText('')
        self.auto_sample_timer.stop()
        self.ui.rob_state.setText('Robot will be stopped')
        self.move_to_cartesian_pose(self.robot_pose_7d)
        self.robot_stopped = True

    def handle_move(self):
        value = self.ui.text.toPlainText()
        if len(value.splitlines()) != 6:
            QMessageBox.about(self.ui, 'Warning', 'Please input translation and rotation value(6 numbers)! ')
            return
        self.transform = list()
        for line in value.splitlines():
            try:
                num = float(line)
                self.transform.append(num)
            except:
                print("Please input pose again!!")

        ## Move the end-effect of the robot, right multiplied by T
        transform_mat = self.tf.angle_to_matrix(self.transform)
        start_mat = self.tf.quat_to_matrix(self.rob_t + self.rob_q)
        desired_mat = np.dot(transform_mat, start_mat)
        self.desired_pose = self.tf.matrix_to_quat(desired_mat)
        if self.robot_stopped:
            self.move_to_cartesian_pose(self.desired_pose)
        else:
            QMessageBox.about(self.ui, 'Warning', 'Please wait to finish the movement! ')
            return

    def handle_compute(self):
        if self.sample_cnt == 0:
            self.ui.warning.setText('Please sample first!')
            return
        cali_mat, notes = self.compute.compute(self.sample_path, method=self.cali_method)
        
        self.ui.warning.setText(notes)
        if len(cali_mat) != 0:
            self.sample.save_cali_res_txt(cali_mat, self.result_path)
            cali_mat = np.round(cali_mat, decimals=4)
            self.ui.result.setPlainText(f'result: \n{cali_mat[0][0]}, {cali_mat[0][1]}, {cali_mat[0][2]}, {cali_mat[0][3]}\n{cali_mat[1][0]}, {cali_mat[1][1]}, {cali_mat[1][2]}, {cali_mat[1][3]}\n{cali_mat[2][0]}, {cali_mat[2][1]}, {cali_mat[2][2]}, {cali_mat[2][3]}\n{cali_mat[3][0]}, {cali_mat[3][1]}, {cali_mat[3][2]}, {cali_mat[3][3]}')
        else:
            QMessageBox.about(self.ui, 'Warning', 'Check sample! ')

    def handle_reset(self, is_auto=False):
        self.sample_nums = 0
        self.ui.result.setPlainText('result:')

        if is_auto:
            return
        model = QStandardItemModel()
        self.sample_cnt = 0
        self.range_sample_cnt = 0
        self.ui.sample_range_count.setText('Sample times: ' + '0')
        self.ui.sample_count.setText('Sample times: ' + '0')
        self.ui.text.setPlainText('0' + '\n' + '0' + '\n' + '0' + '\n' + '0' + '\n' + '0' + '\n' + '0')
        self.ui.range_sample_view.setModel(model)
        self.ui.range_view.setModel(model)
        self.range_sample.save_sample_init()
        self.ui.rob_table_view.setModel(model)
        self.ui.cam_table_view.setModel(model)
        self.compute.reset_sample_buffer()
        self.sample.save_sample_init()

    def auto_sample_single_task(self):
        if self.sample_cnt == self.sample_nums:
            self.ui.warning.setText('The auto sample is finished!')
            self.auto_sample_timer.stop()
        ## To make sure the robot is stopped, detect one with interval 1s.
        stopped1 = self.robot_stopped
        for i in range(20):
            QApplication.processEvents()
            time.sleep(0.05)
        stopped2 = self.robot_stopped
        if not (stopped1 and stopped2):
            return
        loop_x_pose, notes = self.loopx.get_pose_loopx()
        if np.sum(loop_x_pose) == 1:
            self.ui.warning.setText('Warning!!! Please check tracking camera data!')
            print('Warning!!! The camera data is unnormal, please stop and check the camera! ')
            return
        else:
            self.ui.warning.setText('')
        self.sample.save_sample(self.rob_t, self.rob_q, loop_x_pose[:3], loop_x_pose[3:])
        self.show_sample()
        self.desired_pose = self.sample.random_sample_pose(self.max_value, self.min_value)
        diff = np.sum(np.abs(self.robot_pose_7d - self.desired_pose))

        while diff < 0.0005 or diff > 1:
            self.desired_pose = self.sample.random_sample_pose(self.max_value, self.min_value)
            diff = np.sum(np.abs(self.robot_pose_7d - self.desired_pose))

        self.move_to_cartesian_pose(self.desired_pose)
        self.sample_cnt += 1
        self.ui.sample_count.setText('Sample times: ' + str(self.sample_cnt))
           
    def show_range_sample(self):
        ee_to_base = self.range_sample.get_sample()
        ee_to_base_angle_mm = self.tf.pose_7d_to_pose_6d(np.array(ee_to_base), True)
        if len(ee_to_base_angle_mm) == 0:
            return
        range_table_model = QStandardItemModel()
        range_table_model.setHorizontalHeaderLabels(['rob_tx(mm)', 'rob_ty(mm)', 'rob_tz(mm)', 'rob_rx(deg)', 'rob_ry(deg)', 'rob_rz(deg)'])
        for value in ee_to_base_angle_mm:
            range_table_model.appendRow([QStandardItem(str(entry)) for entry in value])
        self.ui.range_sample_view.setModel(range_table_model)

    def show_sample(self):
        ee_to_base_angle, tar_to_cam_angle = self.sample.get_sample()
        if len(ee_to_base_angle) == 0 and len(tar_to_cam_angle) == 0:
            return
        rob_table_model = QStandardItemModel()
        rob_table_model.setHorizontalHeaderLabels(['rob_tx(mm)', 'rob_ty(mm)', 'rob_tz(mm)', 'rob_rx(deg)', 'rob_ry(deg)', 'rob_rz(deg)'])
        for value in ee_to_base_angle:
            rob_table_model.appendRow([QStandardItem(str(entry)) for entry in value])
        self.ui.rob_table_view.setModel(rob_table_model)

        cam_table_model = QStandardItemModel()
        cam_table_model.setHorizontalHeaderLabels(['cam_tx(mm)', 'cam_ty(mm)', 'cam_tz(mm)', 'cam_rx(deg)', 'cam_ry(deg)', 'cam_rz(deg)'])
        for value in tar_to_cam_angle:
            cam_table_model.appendRow([QStandardItem(str(entry)) for entry in value])
        self.ui.cam_table_view.setModel(cam_table_model)

    def robot_motion_detection(self, event):
        if self.robot_pose_7d is None or self.desired_pose is None:
            return
        pose_reached = bool(np.sum(np.abs(self.robot_pose_7d - self.desired_pose)) < 0.0005)
        ## TODO:
        self.robot_stopped = pose_reached
        if not self.robot_stopped:
            self.ui.rob_state.setText('Robot is moving!')
        else:
            self.ui.rob_state.setText('Robot is stopped')

    def move_to_cartesian_pose(self, desired_pose):
        posemsg = PoseStamped()
        posemsg.header.frame_id = "iiwa_link_0"

        posemsg.pose.position.x = desired_pose[0]
        posemsg.pose.position.y = desired_pose[1]
        posemsg.pose.position.z = desired_pose[2]
        posemsg.pose.orientation.x = desired_pose[3]
        posemsg.pose.orientation.y = desired_pose[4]
        posemsg.pose.orientation.z = desired_pose[5]
        posemsg.pose.orientation.w = desired_pose[6]

        self.command_pub.publish(posemsg)

if __name__ == '__main__':
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication([])
    project_path = os.path.dirname(os.path.dirname(__file__))
    np.set_printoptions(precision=6, suppress=True)
    cali = Calibration(project_path)
    cali.ui.show()
    sys.exit(app.exec())