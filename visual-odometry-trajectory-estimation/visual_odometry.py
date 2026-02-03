#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os
from typing import List, Tuple, Optional
import time
from scipy.signal import savgol_filter

class Frame:
    def __init__(self, image_path: str, frame_id: int):
        self.id = frame_id
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.keypoints = None
        self.descriptors = None
        self.pose = np.eye(4)  #4x4 transformation matrix
        self.rotation_matrix = np.eye(3)
        self.translation_vector = np.zeros((3, 1))
        self.processed = False
        self.feature_type = None
        self.timestamp = time.time()
        
    def extract_features(self, detector, feature_type):
        self.keypoints, self.descriptors = detector.detectAndCompute(self.gray, None)
        self.processed = True
        self.feature_type = feature_type
        if self.descriptors is not None:
            if feature_type == "sift":
                self.descriptors = self.descriptors.astype(np.float32)
            else:  # "orb"
                self.descriptors = self.descriptors.astype(np.uint8)
        
    def update_pose(self, rotation: np.ndarray, translation: np.ndarray):
        self.rotation_matrix = rotation.copy()
        self.translation_vector = translation.copy()
        
        #updating 4x4 transformation matrix
        self.pose[:3, :3] = rotation
        self.pose[:3, 3:4] = translation


class VisualOdometry:
    def __init__(self, dataset_path: str, use_sift: bool = False):
        self.dataset_path = dataset_path
        self.frames: List[Frame] = []
        self.trajectory = []  #list of camera positions
        self.use_sift = use_sift
        self.feature_type = "sift" if use_sift else "orb"
        
        #Feature detector
        if use_sift:
            self.detector = cv2.SIFT_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.detector = cv2.ORB_create(nfeatures=3000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        #The camera intrinsic matrix that will be estimated from image dimensions
        self.K = None

        self.T_coord = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ], dtype=np.float64)

        #current pos
        self.current_pose = np.eye(4)
        self.current_rotation = np.eye(3)
        self.current_translation = np.zeros((3, 1))
        
    def load_images(self):
        image_files = sorted(glob.glob(os.path.join(self.dataset_path, "*.jpg")) + 
                           glob.glob(os.path.join(self.dataset_path, "*.png")))
        
        if not image_files:
            raise ValueError(f"No images found in {self.dataset_path}")
        
        print(f"Loading {len(image_files)} images...")
        
        for idx, img_path in enumerate(image_files):
            frame = Frame(img_path, idx)
            self.frames.append(frame)
            
        #Estimating the camera intrinsic matrix from the first image
        first_frame = self.frames[0]
        h, w = first_frame.gray.shape
        self.K = self._estimate_intrinsic_matrix(w, h)
        
        print(f"Loaded {len(self.frames)} frames")
        print(f"Image size: {w}x{h}")
        print(f"Camera matrix K:\n{self.K}")
        
    def _estimate_intrinsic_matrix(self, width: int, height: int) -> np.ndarray:
        #assuming focal length is approximately image width
        focal_length = width
        cx = width / 2.0
        cy = height / 2.0
        
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return K
    
    def match_features(self, frame1: Frame, frame2: Frame, 
                      ratio_threshold: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        
        if frame1.descriptors is None or frame2.descriptors is None:
            print("Warning: No descriptors found in one of the frames")
            return np.array([]), np.array([])
        
        #match descriptors
        matches = self.matcher.knnMatch(frame1.descriptors, frame2.descriptors, k=2)
        
        #applying ratio test (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 8:
            print(f"Warning: Only {len(good_matches)} good matches found")
            return np.array([]), np.array([])
        
        #extracting matched keypoint coordinates
        pts1 = np.float32([frame1.keypoints[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([frame2.keypoints[m.trainIdx].pt for m in good_matches])
        
        return pts1, pts2
    
    def compute_relative_pose(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #computing Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, 
                                       method=cv2.RANSAC, 
                                       prob=0.999, 
                                       threshold=1.0)
        
        if E is None:
            print("Warning: Essential matrix computation failed")
            return np.eye(3), np.zeros((3, 1))
        
        #using the mask for filtering the points
        pts1_filtered = pts1[mask.ravel() == 1]
        pts2_filtered = pts2[mask.ravel() == 1]
        
        #recover pose
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_filtered, pts2_filtered, self.K)
        
        return R, t
    
    def process_frame_pair(self, frame1: Frame, frame2: Frame):
        if (not frame1.processed) or (frame1.feature_type != self.feature_type):
            frame1.extract_features(self.detector, self.feature_type)

        if (not frame2.processed) or (frame2.feature_type != self.feature_type):
            frame2.extract_features(self.detector, self.feature_type)
        
        #match features
        pts1, pts2 = self.match_features(frame1, frame2)
        
        if len(pts1) < 8:
            print(f"Insufficient matches between frames {frame1.id} and {frame2.id}")
            frame2.update_pose(self.current_rotation, self.current_translation) #keep previous pose
            return
        
        R_rel, t_rel = self.compute_relative_pose(pts1, pts2) #computing relative pose
        
        # Transform coordinate system to desired (X-right, Y-forward, Z-up)
        R_rel = self.T_coord @ R_rel @ self.T_coord.T
        t_rel = self.T_coord @ t_rel
        
        # updating accumulate pose by: New pose = Current pose * Relative pose
        self.current_translation = self.current_translation + self.current_rotation @ t_rel
        self.current_rotation = R_rel @ self.current_rotation
        
        #updating frame pose
        frame2.update_pose(self.current_rotation, self.current_translation)
        
        #storing position in trajectory
        position = self.current_translation.flatten()
        self.trajectory.append(position.copy())
        
        print(f"Frame {frame2.id}: Position = [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
    
    def run(self):
        if len(self.frames) < 2:
            raise ValueError("Need at least 2 frames to run visual odometry")
        
        print("\nStarting Visual Odometry processing...")
        print("=" * 60)
        
        #initializing first frame at origin
        self.frames[0].extract_features(self.detector)
        self.trajectory.append(np.zeros(3))
        
        #processing consecutive frame pairs
        for i in range(len(self.frames) - 1):
            print(f"\nProcessing frames {i} -> {i+1}")
            self.process_frame_pair(self.frames[i], self.frames[i+1])
        
        print("\n" + "=" * 60)
        print("Visual Odometry processing complete!")
        print(f"Total frames processed: {len(self.frames)}")
        print(f"Trajectory length: {len(self.trajectory)}")
        
    def get_trajectory(self) -> np.ndarray:
        return np.array(self.trajectory)
    
    def get_frame_with_keypoints(self, frame_idx: int) -> np.ndarray:
        if frame_idx >= len(self.frames):
            return None
        
        frame = self.frames[frame_idx]
        img_with_kp = cv2.drawKeypoints(frame.image, frame.keypoints, None, 
                                        color=(0, 255, 0), 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img_with_kp

    def smooth_trajectory(self, window_length=11, polyorder=3):
        if len(self.trajectory) < window_length:
            print(f"Warning: Trajectory too short for smoothing (need at least {window_length} points)")
            return np.array(self.trajectory)
        
        trajectory_array = np.array(self.trajectory)
        
        smoothed_trajectory = np.zeros_like(trajectory_array)
        smoothed_trajectory[:, 0] = savgol_filter(trajectory_array[:, 0], window_length, polyorder)
        smoothed_trajectory[:, 1] = savgol_filter(trajectory_array[:, 1], window_length, polyorder)
        smoothed_trajectory[:, 2] = savgol_filter(trajectory_array[:, 2], window_length, polyorder)
        
        return smoothed_trajectory