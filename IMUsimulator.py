'''
This file is to generate the IMU for the whole dataset
'''
import os
import sys, argparse

from os.path import join, isdir
from os import mkdir
import numpy as np

from scipy import interpolate
from scipy.spatial.transform import Rotation, RotationSpline
import yaml

import pypose as pp
import torch
import pypose.module as pm
import matplotlib.pyplot as plt

def interpolate_translation(time, data, ips=100):
    time_interpolate = np.arange(round(time.max() * ips)) / ips
    pose = []
    vel = []
    accel = []

    for i in range(3):
        x = data[:,i]
        tck = interpolate.splrep(time, x, s = 0, k = 4)
        x_new = interpolate.splev(time_interpolate, tck, der=0)
        vel_new = interpolate.splev(time_interpolate, tck, der = 1)
        accel_new = interpolate.splev(time_interpolate, tck, der = 2)
        pose.append(x_new)
        vel.append(vel_new)
        accel.append(accel_new)
    accel = np.array(accel).T
    vel = np.array(vel).T
    pose = np.array(pose).T
    return time_interpolate, accel, vel, pose

def interpolate_rotation(time, data, ips = 100):
    rotations = Rotation.from_quat(data)
    spline = RotationSpline(time, rotations)

    time_interpolate = np.arange(round(time.max() * ips)) / ips
    angles = spline(time_interpolate).as_euler('xyz', degrees=False) #XYZ
    gyro= spline(time_interpolate, 1)
    angular_acceleration = spline(time_interpolate, 2)

    return time_interpolate, angular_acceleration, gyro, angles

def interpolate_rotation_debug(time, data, ips = 100):
    rotations = Rotation.from_quat(data)
    spline = RotationSpline(time, rotations)

    time_interpolate = np.arange(round(time.max() * ips)) / ips
    angles = spline(time_interpolate).as_euler('zyx', degrees=False)
    angular_rate = spline(time_interpolate, 1)
    angular_acceleration = spline(time_interpolate, 2)

    _transform = np.array([np.array([[1, 0,                -np.sin(angle[1])],
                            [0, np.cos(angle[0]),  np.sin(angle[0])*np.cos(angle[1])],
                            [0, -np.sin(angle[0]), np.cos(angle[0])*np.cos(angle[1])]]) for angle in angles])
    gyro = np.einsum("bij, bj -> bi", _transform, angular_rate)

    return time_interpolate, angular_acceleration, angular_rate, gyro, angles

def interpolate_traj(time, data, gravity = None, ips = 100):
    if gravity is None:
        gravity = np.zeros((3,))

    time_interpolate, accel, vel, pose = interpolate_translation(time,data[:,:3],ips=ips)

    time_interpolate, angular_accel, rate, angles = interpolate_rotation(time,data[:,3:],ips=ips)
    rotations = Rotation.from_euler("xyz", angles, degrees=False)
    angle_Mat = rotations.as_matrix()

    accel_body = np.matmul(np.expand_dims(accel+gravity,1), angle_Mat).squeeze(1)
    vel_body = np.matmul(np.expand_dims(vel,1),angle_Mat).squeeze(1)

    accel_body_nograv = np.matmul(np.expand_dims(accel,1), angle_Mat).squeeze(1)
    
    return time_interpolate, accel_body, vel, pose, rate, angles, vel_body, accel, accel_body_nograv

def generate_imudata(posefile, outputdir, gt_fps = 10, imu_fps = 100):
    """
    """
    gravity = np.array([0, 0, 9.8])
    poses = np.loadtxt(posefile,dtype = np.float32)
    length = poses.shape[0]
    img_time = np.float32(np.arange(length))/gt_fps
    
    # Fit data
    imu_time, accel_body, vel, pose, rate, angles, vel_body, accel_nograv, accel_nograv_body = interpolate_traj(img_time, poses, gravity = gravity, ips = imu_fps)

    if not isdir(outputdir):
        mkdir(outputdir)

    np.savetxt(join(outputdir,"acc.txt"),accel_body)
    np.savetxt(join(outputdir,"gyro.txt"),rate)
    np.savetxt(join(outputdir,"imu_time.txt"),imu_time)
    np.savetxt(join(outputdir,"cam_time.txt"),img_time)
    np.savetxt(join(outputdir,"vel_global.txt"), vel)
    np.savetxt(join(outputdir,"vel_body.txt"), vel_body)
    np.savetxt(join(outputdir,"pos_global.txt"), pose)
    np.savetxt(join(outputdir,"ori_global.txt"), angles)
    np.savetxt(join(outputdir,"acc_nograv.txt"), accel_nograv)
    np.savetxt(join(outputdir,"acc_nograv_body.txt"), accel_nograv_body)

    np.save(join(outputdir,"acc"),accel_body)
    np.save(join(outputdir,"gyro"),rate)
    np.save(join(outputdir,"imu_time"),imu_time)
    np.save(join(outputdir,"cam_time"),img_time)
    np.save(join(outputdir,"vel_global"), vel)
    np.save(join(outputdir,"vel_body"), vel_body)
    np.save(join(outputdir,"pos_global"), pose)
    np.save(join(outputdir,"ori_global"), angles)
    np.save(join(outputdir,"acc_nograv"), accel_nograv)
    np.save(join(outputdir,"acc_nograv_body"), accel_nograv_body)

    with open(join(outputdir, 'parameter.yaml'), 'w') as f:
        params = {'img_fps': gt_fps, 'imu_fps': imu_fps}
        yaml.dump(params, f)
    
# example:
# python -m postprocessing.imu_generator 
#   --data-root /home/amigo/tmp/test_root 
#   --env-folders downtown2

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default="/Users/pro/project/data/P001", help='root directory of the dataset')
    parser.add_argument('--imufps', type=int, default=200, help='imu frequency')
    parser.add_argument('--gtfps', type=int, default=10, help='gt frequency')
    parser.add_argument('--gtrot', default=False, action = 'store_true', help='use gt rotation')

    args = parser.parse_args()

    pose_gt = np.loadtxt(join(args.dataroot, 'pose_left.txt'))
    
    length = pose_gt.shape[0]
    print(length)

    gt_time = np.float32(np.arange(length))/args.gtfps
    time_interpolate, angular_acceleration, gyro, angles = interpolate_rotation(gt_time, pose_gt[:,3:],ips=args.imufps)
    time_interpolate, accel, vel, pose = interpolate_translation(gt_time, pose_gt[:,:3],ips=args.imufps)
    gravity = np.array([0, 0, 9.81])

    rotations = Rotation.from_euler("xyz", angles, degrees=False)
    angle_Mat = rotations.as_matrix()

    ROTSO3 = pp.SO3(rotations.as_quat())

    accel_body = np.matmul(np.expand_dims(accel+gravity,1), angle_Mat).squeeze(1)
    vel_body = np.matmul(np.expand_dims(vel,1),angle_Mat).squeeze(1)

    # accel_body_nograv = np.matmul(np.expand_dims(accel,1), angle_Mat).squeeze(1)

    print('*** IMU generation start ***')
    Int = pm.IMUPreintegrator(gravity = 9.81)
    init = {
        'rot': ROTSO3[0],
        'vel': torch.tensor(vel[0]),
        'pos': torch.tensor(pose[0]),
    }
    length = int((pose_gt.shape[0]-1) * args.imufps / args.gtfps)
    dt = torch.ones(length,1) * (1.0 / args.imufps)
    acc = torch.tensor(accel_body).float()
    gyro = torch.tensor(gyro).float()

    if args.gtrot:
        out_state = Int(init_state = init, dt = dt, acc = acc, gyro = gyro, rot = ROTSO3)
    else:
        out_state = Int(init_state = init, dt = dt, acc = acc, gyro = gyro)
    pos_diff = (out_state['pos'] - torch.tensor(pose)).norm(dim=-1).numpy()
    plt.plot(out_state['pos'][0,:,0], out_state['pos'][0,:,1], label = 'imu')
    plt.plot(pose[:,0], pose[:,1], label = 'gt')
    plt.legend()
    vel_diff = (out_state['vel'][0, :,0] - torch.tensor(vel[:,0])).numpy()
    diff = ((ROTSO3.Inv() * out_state['rot']).Log()* 180/np.pi).numpy()
    plt.show()

