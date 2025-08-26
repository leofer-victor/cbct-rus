#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   calibration_utils.py
@Time    :   2024-07-10 13:27:19
@Author  :   feng li
@Contact :   feng.li@tum.de
@Description    :   
'''

import cv2
import numpy as np
import math
import socket
import yaml
import time
import torch
import vtkmodules.all as vtk
import matplotlib.pyplot as plt
import sys

from vtkmodules.vtkRenderingCore import vtkRenderer, vtkColorTransferFunction, vtkImageSlice, vtkImageSliceMapper, vtkImageActor
sys.path.append('/home/leofer/ros_projects/loop_x/src/loop_x/sam')
from os import path, listdir
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from transforms3d import euler, quaternions
# from transforms3d import quaternions
from scipy.ndimage import label, center_of_mass
from scipy.optimize import minimize

class RangeSampler():
    def __init__(self, sample_file_path):
        if sample_file_path is None:
            return
        self.sample_file_path = sample_file_path
        self.tf = Transform()

    def save_sample_init(self):
        with open(self.sample_file_path, 'w') as f:
            f.write('rob_tx'+'\t' + 'rob_ty'+'\t' + 'rob_tz'+'\t' +
                    'rob_qx'+'\t' + 'rob_qy'+'\t' + 'rob_qz'+'\t' + 'rob_qw'+'\n')
            
    def save_sample(self, rob_t, rob_q):
        rob_t = np.round(np.array(rob_t), decimals=6)
        rob_q = np.round(np.array(rob_q), decimals=6)
        with open(self.sample_file_path, 'a') as f:
            for rx in rob_t:
                f.write(str(rx)+'\t')
            for idx, rq in enumerate(rob_q):
                if idx == 3:
                    f.write(str(rq)+'\n')
                else:
                    f.write(str(rq)+'\t')

    def get_sample(self):
        with open(self.sample_file_path, 'r') as file:
            txt_list = file.readlines()[1:]
            rob_pose = []
            for txt_line in txt_list:
                txt_line = txt_line.strip()
                txt_line = txt_line[0:-1]
                sample_line = [float(entry) for entry in txt_line.split('\t')]
                rob_pose.append(sample_line)
        return rob_pose

class CaliSampler():
    def __init__(self, sample_file_path, result_filepath):
        self._result_filepath = result_filepath
        self.results_path = '/home/leofer/experiment/loop_x/data/results/loopx_movement'

        if sample_file_path is None:
            return
        self.sample_file_path = sample_file_path
        self.tf = Transform()

    def save_cali_res_txt(self, res_mat,
                        filepath,
                        parent_frame_id="kuka_iiwa",
                        child_frame_id="camera"
                        ):
        with open(filepath, 'w') as file:
            num_row = 0
            for row in range(4):
                for col in range(4):
                    value = str(res_mat[row, col])
                    if num_row < 3:
                        file.write(value + '\t')
                        num_row += 1
                    else:
                        file.write(value + '\n')
                        num_row = 0
            file.write('parent_frame_id: ' + parent_frame_id + '\n' +
                    'child_frame_id: ' + child_frame_id + '\n') 

    def save_sample_init(self):
        with open(self.sample_file_path, 'w') as f:
            f.write('rob_tx'+'\t' + 'rob_ty'+'\t' + 'rob_tz'+'\t' +
                    'rob_qx'+'\t' + 'rob_qy'+'\t' + 'rob_qz'+'\t' + 'rob_qw'+'\t' +
                    'cam_tx'+'\t' + 'cam_ty'+'\t' + 'cam_tz'+'\t' +
                    'cam_qx'+'\t' + 'cam_qy'+'\t' + 'cam_qz'+'\t' + 'cam_qw'+'\t' + '\n')

    def save_sample(self, rob_t, rob_q, cam_t, cam_q):
        rob_t = np.round(np.array(rob_t), decimals=5)
        rob_q = np.round(np.array(rob_q), decimals=5)
        cam_t = np.round(np.array(cam_t), decimals=5)
        cam_q = np.round(np.array(cam_q), decimals=5)
        with open(self.sample_file_path, 'a') as f:
            for rx in rob_t:
                f.write(str(rx)+'\t')
            for rq in rob_q:
                f.write(str(rq)+'\t')
            
            if len(cam_t) == 0:
                f.write('\n')
            for ct in cam_t:
                f.write(str(ct)+'\t')
            for idx, cq in enumerate(cam_q):
                if idx == 3:
                    f.write(str(cq)+'\n')
                else:
                    f.write(str(cq)+'\t')

    def save_transformation(self, res_mat, filepath):
        with open(filepath, 'w') as file:
            num_row = 0
            for row in range(4):
                for col in range(4):
                    value = str(res_mat[row, col])
                    if num_row < 3:
                        file.write(value + '\t')
                        num_row += 1
                    else:
                        file.write(value + '\n')
                        num_row = 0

    def save_sample_pose(self, rob_t, rob_q, cam_t, cam_q, index=0):
        rob_t = np.round(np.array(rob_t), decimals=6)
        rob_q = np.round(np.array(rob_q), decimals=6)
        cam_t = np.round(np.array(cam_t), decimals=6)
        cam_q = np.round(np.array(cam_q), decimals=6)

        ee_to_base_r_mat = quaternions.quat2mat(self.tf.xyzw_to_wxyz(rob_q))
        tar_to_cam_r_mat = quaternions.quat2mat(self.tf.xyzw_to_wxyz(cam_q))

        ee_to_base = np.identity(4)
        ee_to_base[:3, :3] = ee_to_base_r_mat
        ee_to_base[:3, 3] = rob_t

        rob_name = 'rob_{:04}'.format(index) + '.txt'
        rob_pose_path = path.join(self.results_path, rob_name) 
        self.save_transformation(ee_to_base, rob_pose_path)

        tar_to_cam = np.identity(4)
        tar_to_cam[:3, :3] = tar_to_cam_r_mat
        tar_to_cam[:3, 3] = cam_t

        cam_name = 'cam_{:04}'.format(index) + '.txt'
        cam_pose_path = path.join(self.results_path, cam_name) 
        self.save_transformation(tar_to_cam, cam_pose_path)
    
    def read_and_compute_pose(self):
        cam_name1 = 'cam_{:04}'.format(0) + '.txt'
        cam_pose_path1 = path.join(self.results_path, cam_name1) 
        cam_pose1 = self.get_pose_t(cam_pose_path1)

        cam_name2 = 'cam_{:04}'.format(1) + '.txt'
        cam_pose_path2 = path.join(self.results_path, cam_name2) 
        cam_pose2 = self.get_pose_t(cam_pose_path2)

        t_cam = np.dot(cam_pose2, np.linalg.inv(cam_pose1))

        rob_name1 = 'rob_{:04}'.format(0) + '.txt'
        rob_pose_path1 = path.join(self.results_path, rob_name1) 
        rob_pose1 = self.get_pose_t(rob_pose_path1)

        rob_name2 = 'rob_{:04}'.format(1) + '.txt'
        rob_pose_path2 = path.join(self.results_path, rob_name2) 
        rob_pose2 = self.get_pose_t(rob_pose_path2)

        t_rob = np.dot(rob_pose2, np.linalg.inv(rob_pose1))
        return t_cam, t_rob

    def get_pose_t(self, filepath):
        pose_t = np.identity(4)
        with open(filepath, 'r') as file:
            txt_list = file.readlines()
            if len(txt_list) != 4:
                return
            for index, txt_line in enumerate(txt_list):
                if index == 4:
                    break
                txt_line = txt_line[0:-1]
                line = [float(entry) for entry in txt_line.split('\t')]
                pose_t[index] = line
        return pose_t

    def random_r(self, max_value, min_value):
        res = 0
        if max_value - min_value > 180:
            low = max_value - 180
            high = min_value + 180
            ran_r = np.random.random() * (high - low) + low
            if ran_r >= 0:
                res = ran_r - 180
            else:
                res = ran_r + 180
        else:
            res = np.random.random() * (max_value - min_value) + min_value
        return res
    
    def random_sample_pose(self, max_value, min_value):
        range_value = np.array(max_value) - np.array(min_value)
        tx = np.random.random() * range_value[0] + min_value[0]
        ty = np.random.random() * range_value[1] + min_value[1]
        tz = np.random.random() * range_value[2] + min_value[2]

        rx = self.random_r(max_value[3], min_value[3])
        ry = self.random_r(max_value[4], min_value[4])
        rz = self.random_r(max_value[5], min_value[5])

        pose_angle = np.array([tx, ty, tz, rx, ry, rz])
        pose = self.tf.angle_to_quat(pose_angle)
        return pose

    def get_sample(self):
        with open(self.sample_file_path, 'r') as file:
            txt_list = file.readlines()[1:]
            sample_list = []
            for txt_line in txt_list:
                txt_line = txt_line.strip()
                sample_line = [float(entry) for entry in txt_line.split('\t')]
                sample_list.append(sample_line)
                
        sample = np.array(sample_list)
        ee_to_base = sample[:, :7]
        tar_to_cam = sample[:, 7:]

        if ee_to_base.shape != tar_to_cam.shape:
            print("Sample warning, the number of the robot pose doesn't equal to the number of the camera pose!")
            if ee_to_base.shape[1] == 7:
                ee_to_base_angle = self.tf.pose_7d_to_pose_6d(ee_to_base, True)
                return ee_to_base_angle, []
            elif tar_to_cam.shape[1] == 7:
                tar_to_cam_angle = self.tf.pose_7d_to_pose_6d(tar_to_cam, True)
                return [], tar_to_cam_angle
            else:
                return [], []
        
        ee_to_base_angle = self.tf.pose_7d_to_pose_6d(ee_to_base, True)
        tar_to_cam_angle = self.tf.pose_7d_to_pose_6d(tar_to_cam, True)

        return ee_to_base_angle, tar_to_cam_angle
      
class CaliCompute():
    def __init__(self):
        self.AVAILABLE_ALGORITHMS = {
            'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
            'Park': cv2.CALIB_HAND_EYE_PARK,
            'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
            'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        self.tf = Transform()
        self.reset_sample_buffer()

    def __load_sample_buffer_from_file(self, filepath: str):
        with open(filepath, 'r') as file:
            txt_list = file.readlines()[1:]
            sample_list = []
            for txt_line in txt_list:
                txt_line = txt_line.strip()
                sample_line = [float(entry) for entry in txt_line.split()]
                if len(sample_line) != 14:
                    return False
                sample_list.append(sample_line)
        for s in np.array(sample_list):
            self.__push_back_sample_buffer(s[:3], s[3:7], s[7:10], s[10:14])
        
        return True

    def __push_back_sample_buffer(self, ee_to_base_t: list,
                                ee_to_base_q: list,
                                tar_to_cam_t: list,
                                tar_to_cam_q: list):

        ee_to_base_r_mat = quaternions.quat2mat(self.tf.xyzw_to_wxyz(ee_to_base_q))
        tar_to_cam_r_mat = quaternions.quat2mat(self.tf.xyzw_to_wxyz(tar_to_cam_q))

        ee_to_base = np.identity(4)
        ee_to_base[:3, :3] = ee_to_base_r_mat
        ee_to_base[:3, 3] = ee_to_base_t
        self.ee_to_base_mat.append(ee_to_base)

        tar_to_cam = np.identity(4)
        tar_to_cam[:3, :3] = tar_to_cam_r_mat
        tar_to_cam[:3, 3] = tar_to_cam_t
        self.tar_to_cam_mat.append(tar_to_cam)

    def reset_sample_buffer(self):
        self.ee_to_base_mat = []
        self.tar_to_cam_mat = []

    def compute(self, file_path, method='Tsai-Lenz', is_eye_to_hand=True):
        res = np.identity(4)
        np.set_printoptions(suppress=True)
        notes = ''
        if not self.__load_sample_buffer_from_file(file_path):
            notes = 'Please check the sample data of robot and loop x!'
            return [], notes
        if len(self.ee_to_base_mat) < 5:
            notes = 'Sampling more!'
            return [], notes

        if is_eye_to_hand:
            base_to_ee_mat = []
            for mat in self.ee_to_base_mat:
                mat = np.linalg.inv(mat)
                base_to_ee_mat.append(mat)

        base_to_ee_mat = np.array(base_to_ee_mat)
        tar_to_cam_mat = np.array(self.tar_to_cam_mat)

        cali_r_mat, cali_t = cv2.calibrateHandEye(base_to_ee_mat[:, :3, :3], base_to_ee_mat[:, :3, 3],
                                                  tar_to_cam_mat[:, :3, :3], tar_to_cam_mat[:, :3, 3],
                                                  method=self.AVAILABLE_ALGORITHMS[method])
        res[:3, :3] = cali_r_mat
        res[:3, 3] = cali_t.reshape(-1)
        print('Calibration results: \n', res, '\n')
        
        return res, notes
    
    def get_calibration_res(self, filepath):
        calibration_res = np.identity(4)
        with open(filepath, 'r') as file:
            txt_list = file.readlines()
            if len(txt_list) != 6:
                return
            for index, txt_line in enumerate(txt_list):
                if index == 4:
                    break
                txt_line = txt_line[0:-1]
                line = [float(entry) for entry in txt_line.split('\t')]
                calibration_res[index] = line
        return calibration_res
    
    def get_coordinate(self, filepath):
        loop_x = np.identity(4)
        with open(filepath, 'r') as file:
            txt_list = file.readlines()
            if len(txt_list) != 6:
                return
            for index, txt_line in enumerate(txt_list):
                if index == 4:
                    break
                txt_line = txt_line[0:-1]
                line = [float(entry) for entry in txt_line.split(' ')]
                loop_x[index] = line
        return loop_x
    
class LoopX():
    def __init__(self):
        self.host = '192.168.199.4'  # Change this to the IP address you want to listen on
        self.port = 55598  # Change this to the port you want to listen on
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(0.09)
        # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # No wait_time
        # sock.setblocking(False) # non-block socket
        self.connected = False
        while not self.connected:
            try:
                self.sock.connect((self.host, self.port))
                self.connected = True
            except Exception as e:
                self.notes_connection = f'Error binding to {self.host}:{self.port}!'
                break
    
    def get_pose_loopx(self):
        notes = 'succeed!'
        position = np.array([0,0,0,0,0,0,1])
        if not self.connected:
            return position, self.notes_connection
        try:
            response = self.sock.recv(102400)
        except Exception as e:
            notes = 'LoopX connected, but nothing received!'
            return position, notes
            
        data_str = response.decode()
        try:
            positions = self.find_optitrack_poses(data_str)
        except Exception as e:
            notes = 'LoopX connected and received, but did not get pose from string!'
            return position, notes
        if len(positions) == 0:
            notes = 'LoopX connected and received, but did not get pose from string!'
            return position, notes
        else:
            return positions[-1], notes
    
    def find_optitrack_poses(self, string):
        # List to store the indices of all occurrences
        poses = []

        # Start index for searching
        start_index = 0

        # Find all occurrences of "OptiTrackPositions" in the string
        while True:
            index_start_p = string.find("OptiTrackPositions=", start_index)
            if index_start_p == -1:
                break
            index_start_p = index_start_p + len("OptiTrackPositions=")

            if string.find("|",index_start_p) == -1:
                index_end_p = string.find("\x00",index_start_p)
            else:
                index_end_p = string.find("|",index_start_p)
            if index_end_p == -1:
                break
            numbers = string[index_start_p:index_end_p]
            start_index = index_end_p + 1
            position = np.array(numbers.split(','), dtype=float)/100
            index_start_r = string.find("OptiTrackRotations=", start_index)
            if index_start_r == -1:
                break
            index_start_r = index_start_r + len("OptiTrackRotations=")

            if string.find("|",index_start_r) == -1:
                index_end_r = string.find("\x00",index_start_r)
            else:
                index_end_r = string.find("|",index_start_r)
            if index_end_r == -1:
                break
            numbers = string[index_start_r:index_end_r]
            start_index = index_end_r + 1
            quat = np.array(numbers.split(','), dtype=float)
            first_element = quat[0]  # Remove the first element from the list
            quat = quat[1:]
            quat = np.append(quat, first_element)
            
            pose = np.concatenate((position, quat))
            poses.append(pose)

        return np.array(poses)
    
class Transform():
    def __init__(self):
        pass

    def get_max_min(self, ee_to_base_angle):
        if len(ee_to_base_angle) == 0:
            return [], []
        max_tmp = np.max(ee_to_base_angle, axis=0)
        min_tmp = np.min(ee_to_base_angle, axis=0)
        max_value = np.full(6, -1000.0)
        min_value = np.full(6, 1000.0)
        for i in range(6):
            max_value[i] = max(max_value[i], max_tmp[i])
            min_value[i] = min(min_value[i], min_tmp[i])
        return np.round(max_value, decimals=3), np.round(min_value, decimals=3)

    def fstr(self, astr):
        format_str = str(astr)
        format_str = format_str.strip('[')
        format_str = format_str.strip(']')
        format_str = '\t'.join(format_str.split())
        return format_str

    def pose_7d_to_pose_6d(self, pose_7d, unit_m_to_mm=False):
        length = pose_7d.shape[0]
        pose_6d = np.empty((length, 6))
        for i in range(length):
            angle = self.quat_to_angle(pose_7d[i], unit_m_to_mm)
            pose_6d[i] = angle
        return np.round(pose_6d, decimals=3)

    def matrix_to_quat(self, matrix):
        trans = matrix[:3, 3]
        rot = matrix[:3, :3]
        qua = self.wxyz_to_xyzw(quaternions.mat2quat(rot)) 
        pose = np.empty(7)
        pose[:3] = trans
        pose[3:7] = qua
        return pose

    def matrix_to_angle(self, matrix, unit_m_to_mm=False):
        trans = matrix[:3, 3]
        rot = matrix[:3, :3]
        angle = euler.mat2euler(rot)
        pose = np.empty(6)
        pose[:3] = trans
        if unit_m_to_mm:
            pose[:3] = pose[:3] * 1000
        angle_new = []
        for value in angle:
            value = self.deg(value)
            angle_new.append(value)
        pose[3:6] = angle_new
        return pose

    def angle_to_matrix(self, pose):
        if len(pose) != 6:
            return
        trans = pose[:3]
        angle = pose[3:]
        rot = euler.euler2mat(self.rad(angle[0]), self.rad(angle[1]), self.rad(angle[2]))
        matrix = np.identity(4)
        matrix[:3, 3] = trans
        matrix[:3, :3] = rot
        return matrix

    def angle_to_quat(self, ang):
        if len(ang) != 6:
            return
        trans = ang[: 3]
        angle = ang[3:]
        qua = self.wxyz_to_xyzw(euler.euler2quat(self.rad(angle[0]), self.rad(angle[1]), self.rad(angle[2])))
        return np.append(trans, qua)

    def quat_to_angle(self, qua, unit_m_to_mm=False):
        if len(qua) != 7:
            return 
        trans = qua[: 3]
        qua = self.xyzw_to_wxyz(qua[3:])
        angle = euler.quat2euler(qua)
        pose = np.empty(6)
        pose[:3] = trans
        if unit_m_to_mm:
            pose[:3] = pose[:3] * 1000
        angle_new = []
        for value in angle:
            value = self.deg(value)
            angle_new.append(value)
        pose[3:] = angle_new
        return pose

    def quat_to_matrix(self, pose):
        if len(pose) != 7:
            return
        trans = pose[: 3]
        qua = pose[3:]
        # wxyz
        rot = euler.quat2mat(self.xyzw_to_wxyz(qua))
        matrix = np.identity(4)
        matrix[:3, 3] = trans
        matrix[:3, :3] = rot
        return matrix

    def xyzw_to_wxyz(self, qua):
        if len(qua) != 4:
            return
        return np.append([qua[3]], qua[: 3])

    def wxyz_to_xyzw(self, qua):
        if len(qua) != 4:
            return
        return np.append(qua[1:], [qua[0]])

    def rad(self, degree):
        return degree * math.pi / 180

    def deg(self, radian):
        return radian * 180 / math.pi
    
    def calculate_new_intercept(self, k1, b1, k2, x0):
        y0 = k1 * x0 + b1
        b2 = y0 - k2 * x0
        
        return b2
    
    def rotate_point(self, point, rot_center, angle):
        x = point[0]
        y = point[1]
        x0 = rot_center[0]
        y0 = rot_center[1]
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        x_new = x0 + (x - x0) * cos_theta - (y - y0) * sin_theta
        y_new = y0 + (x - x0) * sin_theta + (y - y0) * cos_theta
    
        return np.array([x_new, y_new])
    
    def calculate_rotation_angle(self, k1, k2):
        tan_theta = (k2 - k1) / (1 + k1 * k2)
        theta = np.arctan(tan_theta)
        return theta
    
    def line_equation_from_points(self, p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        if x1 == x2:
            raise ValueError("x1 == x2")
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        
        return k, b
    
    def distance_between_lines(self, k, b1, b2):
        distance = (b2 - b1) / math.sqrt(1 + k**2)
        return distance
    
    def translation(self, k, distance):
        translation = distance / math.sin(math.tanh(k))
        return translation
    
    def two_lines_angle(self, k1, k2):
        if (1 + k1 * k2 == 0):
            return 90
        tan_theta = (k1 - k2) / (1 + k1 * k2)
        theta = math.atan(tan_theta)
        angle = math.degrees(theta)
        return angle
    
    def add_line_to_file(self, file_name, array):
        if not path.exists(file_name):
            with open(file_name, 'w') as file:
                np.savetxt(file, array[None], fmt='%8f')
        else:
            with open(file_name, 'a') as file:
                np.savetxt(file, array[None], fmt='%8f')
