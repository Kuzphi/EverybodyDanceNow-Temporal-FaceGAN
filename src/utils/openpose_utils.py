import numpy as np
import math
import cv2
from skimage import filters
from scipy import ndimage
from matplotlib import pyplot as plt
# openpose
import sys
sys.path.append('./PoseEstimation')

# from network.post import *
# // Result for BODY_25 (25 body parts consisting of COCO + foot)
# // const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
# //     {0,  "Nose"},
# //     {1,  "Neck"},
# //     {2,  "RShoulder"},
# //     {3,  "RElbow"},
# //     {4,  "RWrist"},
# //     {5,  "LShoulder"},
# //     {6,  "LElbow"},
# //     {7,  "LWrist"},
# //     {8,  "MidHip"},
# //     {9,  "RHip"},
# //     {10, "RKnee"},
# //     {11, "RAnkle"},
# //     {12, "LHip"},
# //     {13, "LKnee"},
# //     {14, "LAnkle"},
# //     {15, "REye"},
# //     {16, "LEye"},
# //     {17, "REar"},
# //     {18, "LEar"},
# //     {19, "LBigToe"},
# //     {20, "LSmallToe"},
# //     {21, "LHeel"},
# //     {22, "RBigToe"},
# //     {23, "RSmallToe"},
# //     {24, "RHeel"},
# //     {25, "Background"}
# // };
def Edge(l, r):
    return list(zip(range(l, r), range(l + 1, r + 1)))

limbs = [   [ 0,  1],
            [ 1,  2],
            [ 2,  3],
            [ 3,  4],
            [ 1,  5],
            [ 5,  6],
            [ 6,  7],
            [ 1,  8],
            [ 8,  9],
            [ 9, 10],
            [10, 11],
            [11, 24],
            [11, 22],
            [22, 23],
            [ 8, 12],
            [12, 13],
            [13, 14],
            [14, 21],
            [14, 19],
            [19, 20],
            [ 0, 15],
            [ 0, 16],
            [15, 17],
            [16, 18]]
hand = [ 
        [ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4],
        [ 0,  5], [ 5,  6], [ 6,  7], [ 7,  8],
        [ 0,  9], [ 9, 10], [10, 11], [11, 12],
        [ 0, 13], [13, 14], [14, 15], [15, 16],
        [ 0, 17], [17, 18], [18, 19], [19, 20]
    ]
face = [Edge(0, 16),
        Edge(17, 21),
        Edge(22, 26),
        Edge(36, 41) + [[36, 41]],
        Edge(42, 47) + [[42, 47]],
        Edge(27,30),
        Edge(31,35),
        Edge(48,59) + [[48,59]] + Edge(60,67) + [[60, 67]]]

limbs = np.array(limbs)
hand  = np.array(hand)
face  = np.array(face)

def remove_noise(img):
    th = filters.threshold_otsu(img)
    bin_img = img > th
    regions, num = ndimage.label(bin_img)
    areas = []
    for i in range(num):
        areas.append(np.sum(regions == i+1))
    img[regions != np.argmax(areas)+1] = 0
    return img


def create_label(shape, joint_list, person_to_joint_assoc):
    label = np.zeros(shape, dtype=np.uint8)
    cord_list = []
    for limb_type in range(17):
        for person_joint_info in person_to_joint_assoc:
            joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)
            if -1 in joint_indices:
                continue
            joint_coords = joint_list[joint_indices, :2]
            coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
            cord_list.append(joint_coords[0])
            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
            polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(label, polygon, limb_type+1)
    return label,cord_list

def get_pose(param, heatmaps, pafs):
    shape = heatmaps.shape[:2]
    # Bottom-up approach:
    # Step 1: find all joints in the image (organized by joint type: [0]=nose,
    # [1]=neck...)
    joint_list_per_joint_type = NMS(param, heatmaps)
    # joint_list is an unravel'd version of joint_list_per_joint, where we add
    # a 5th column to indicate the joint_type (0=nose, 1=neck...)
    joint_list = np.array([tuple(peak) + (joint_type,) for joint_type,
                           joint_peaks in enumerate(joint_list_per_joint_type) for peak in joint_peaks])

    # Step 2: find which joints go together to form limbs (which wrists go
    # with which elbows)
    paf_upsamp = cv2.resize(pafs, shape, interpolation=cv2.INTER_CUBIC)
    connected_limbs = find_connected_joints(param, paf_upsamp, joint_list_per_joint_type)

    # Step 3: associate limbs that belong to the same person
    person_to_joint_assoc = group_limbs_of_same_person(connected_limbs, joint_list)

    # (Step 4): plot results
    label,cord_list = create_label(shape, joint_list, person_to_joint_assoc)

    return label, cord_list

def draw(joint_coords, label, thickness = 4):
    coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
    limb_dir = joint_coords[0, :] - joint_coords[1, :]
    limb_length = np.linalg.norm(limb_dir)
    angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
    polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), thickness), int(angle), 0, 360, 1)
    x = label.copy()
    cv2.fillConvexPoly(x, polygon, 1)
    return x


def create_body_label(shape, joints): 
    shape = (shape[0], shape[1], 24)

    label = np.zeros(shape, dtype=np.uint8)


    for body_type in range(len(limbs)):
        connection = limbs[body_type]
        if min(joints[connection, 2]) < 0.01:
            continue        
        joint_indices = joints[connection, :2].astype(int)
        label[:,:,body_type] = draw(joint_indices[:, :2], label[:,:,body_type])
        # label[:,:,0] = draw(joint_indices[:, :2], label[:,:,0])
    

    return label

def create_face_label(shape , joints): # 
    shape = (shape[0], shape[1], 10)
    label = np.zeros(shape, dtype=np.uint8)

    for connection_type in range(len(face)):
        for connection in face[connection_type]:
            if min(joints[connection, 2]) < 0.01:
                continue
            joint_indices = joints[connection,:2].astype(int)
            label[:,:,0] = draw(joint_indices[:, :2], label[:,:,0], thickness = 1)
            # label[:,:,connection_type] = draw(joint_indices[:, :2], label[:,:,connection_type])
            
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax.imshow(label[:,:,0])
    # plt.show()

    x = label[:,:, 8].copy()
    polygon = cv2.ellipse2Poly(tuple(joints[68,:2].astype(int)), (1, 1), 0, 0, 360, 1) #left eye
    cv2.fillConvexPoly(x, polygon, 1)
    label[:,:, 8] = x

    x = label[:,:, 9].copy()
    polygon = cv2.ellipse2Poly(tuple(joints[69,:2].astype(int)), (1, 1), 0, 0, 360, 1) #right eye
    cv2.fillConvexPoly(x, polygon, 1)
    label[:,:, 9] = x
    return label


def create_hand_label(shape, joints):
    shape = (shape[0], shape[1], 1)
    label = np.zeros(shape, dtype=np.uint8)


    for connection in hand:
        if min(joints[connection, 2]) < 0.01:
            continue
        joint_indices = joints[connection, :2].astype(int)

        label[:,:,0] = draw(joint_indices[:, :2], label[:,:,0], thickness = 1)
    return label

def create_label_full(shape, joints):
    body = create_body_label(shape, joints['pose_keypoints_2d']) 
    left_hand = create_hand_label(shape, joints['hand_left_keypoints_2d'])
    right_hand = create_hand_label(shape, joints['hand_right_keypoints_2d'])
    face = create_face_label(shape, joints['face_keypoints_2d'])
    res = np.concatenate((body, face, left_hand, right_hand), axis = 2)

    # s = res.max(axis = 2)
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax.imshow(s)
    # plt.show()

    return res