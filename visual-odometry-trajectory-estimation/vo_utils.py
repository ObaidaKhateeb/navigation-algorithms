import os
import glob
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


############################### Frame Class ###############################

class Frame():
    # This is a class-level (static) variable that all Frame instances share.
    _keypoint_id_counter = -1

    def __init__(self, id: int, img: np.ndarray, bow=None):
        self.id: int = id                    # The frame id
        self.img: np.ndarray = img.copy()    # The rgb image
        self.bow = bow                       # The bag of words of that image

        self.keypoints: Tuple                # The extracted ORB keypoints
        self.descriptors: np.ndarray         # The extracted ORB descriptors
        self._extract_features()             # Extract ORB features from the image

        self.pose: np.ndarray = None         # The world -> camera pose transformation matrix
        
        self.match: Dict = {}                # The matches between this frame's keypoints and others'

    def set_keyframe(self, is_keyframe: bool):
        self.is_keyframe = is_keyframe

    def set_matches(self, with_frame_id: int, matches: List[cv2.DMatch], match_mask: np.ndarray, match_type: str):
        """Sets matches with another frame"""
        self.match[with_frame_id] = {}
        self.match[with_frame_id]["matches"] = np.array(matches, dtype=object)
        self.match[with_frame_id]["match_type"] = match_type
        self.match[with_frame_id]["match_mask"] = match_mask

        # Default values for the rest
        self.match[with_frame_id]["initialization"] = None
        self.match[with_frame_id]["use_homography"] = None
        self.match[with_frame_id]["inlier_match_mask"] = None
        self.match[with_frame_id]["pose"] = None
        self.match[with_frame_id]["points"] = None

    def triangulate(self, with_frame_id: int, use_homography: bool, inlier_match_mask: np.ndarray, pose: np.ndarray, stage: str):
        """
        Initializes the frame with another frame.
        """
        self.match[with_frame_id]["initialization"] = stage
        self.match[with_frame_id]["use_homography"] = use_homography
        self.match[with_frame_id]["inlier_match_mask"] = inlier_match_mask
        self.match[with_frame_id]["pose"] = pose      

    def get_matches(self, with_frame_id: int):
        """Returns matches with a specfic frame"""
        return self.match[with_frame_id]["matches"]

    def set_pose(self, pose: np.ndarray):
        self.pose = pose
    
    def _extract_features(self):
        # Initialize the ORB detector
        orb = cv2.ORB_create(nfeatures=2000)
        
        # Detect keypoints and compute descriptors
        kp, desc = orb.detectAndCompute(self.img, None)
        
        # Assign a unique class_id to each keypoint
        for k in kp:
            # Increment the class-level counter
            Frame._keypoint_id_counter += 1
            # Assign the keypoint's class_id
            k.class_id = Frame._keypoint_id_counter
        
        self.keypoints = kp
        self.descriptors = desc


############################### Dataset Class ###############################

class Dataset:
    def __init__(self, data_dir, scene, use_dist=False):
        self.data_dir = data_dir
        self.use_dist = use_dist
        self.scene = scene

        self._current_index = 0 

        self._read()
        self._init_calibration()

    def _read(self):
        # Here self.data_dir is the actual image directory, and self.scene is unused.
        images_dir = self.data_dir

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        self._image_paths = []
        for e in exts:
            self._image_paths += glob.glob(os.path.join(str(images_dir), e))
        self._image_paths.sort()

        if len(self._image_paths) == 0:
            raise RuntimeError(f"No images found in: {images_dir}")

        # Dummy timestamps (optional)
        self._times = np.arange(len(self._image_paths), dtype=float)


    def _init_calibration(self):
        # Build a simple K from the first image size
        img0 = cv2.imread(self._image_paths[0], cv2.IMREAD_GRAYSCALE)
        h, w = img0.shape[:2]
        fx = fy = 0.8 * w
        cx = w / 2.0
        cy = h / 2.0
        self._K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]], dtype=float)


    def get(self):
        timestamp = self._times[self._current_index]
        image_path = self._image_paths[self._current_index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        self._current_index += 1
        return timestamp, image

    def finished(self):
        return self._current_index >= len(self._image_paths)
    
    def get_intrinsics(self):
        return self._K
       
    def length(self):
        return len(self._image_paths)

    def log_img(self, img):
        from config import results_dir
        rgb_save_path = results_dir / "img" / f"{self._current_index}_bw.png"
        save_image(img, rgb_save_path)


############################### Feature Matching ###############################

def match_features(q_frame: Frame, t_frame: Frame):

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 1) Match descriptors (KNN)
    matches = bf.knnMatch(q_frame.descriptors, t_frame.descriptors, k=2)

    # 2) Filter matches with your custom filter (lowe ratio, distance threshold, etc.)
    filtered_matches = filter_matches(matches)

    # 3) Create masks indicating whether each keypoint is used in a match
    q_mask = np.zeros(len(q_frame.keypoints), dtype=bool)
    t_mask = np.zeros(len(t_frame.keypoints), dtype=bool)

    for m in filtered_matches:
        q_mask[m.queryIdx] = True
        t_mask[m.trainIdx] = True

    # 4) **Propagate keypoint IDs**  
    propagate_keypoints(t_frame, q_frame, filtered_matches)

    # 5) Store the matches in each t_frame
    q_frame.set_matches(t_frame.id, filtered_matches, q_mask, "query")
    t_frame.set_matches(q_frame.id, filtered_matches, t_mask, "train")

    return filtered_matches

def propagate_keypoints(t_frame: Frame, q_frame: Frame, matches: List[cv2.DMatch]):
    """Merges the keypoint identifiers for the matches features between query and train frames."""
    for m in matches:
        q_kp = q_frame.keypoints[m.queryIdx]
        t_kp = t_frame.keypoints[m.trainIdx]

        # If the train keypoint has no ID, copy from the query keypoint
        if t_kp.class_id < 0:  # or `t_kp.class_id is None`
            t_kp.class_id = q_kp.class_id

        # If the query keypoint has no ID, copy from the train keypoint
        elif q_kp.class_id <= 0:
            q_kp.class_id = t_kp.class_id

        # If both have IDs but they differ, pick a strategy (e.g., overwrite one)
        elif q_kp.class_id != t_kp.class_id:
            # Naive approach: unify by assigning query ID to train ID
            # or vice versa. Real SLAM systems often handle merges in a global map.
            t_kp.class_id = q_kp.class_id

def filter_matches(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


############################### Pose Estimation ###############################

def estimate_pose(q_frame: Frame, t_frame: Frame, K: np.ndarray):
    # Extract the matches between the previous and current frame
    matches = q_frame.get_matches(t_frame.id)
    num_matches = len(matches)
    if len(matches) < 5:
        print("Not enough matches to compute the Essential Matrix!")
        return None, False

    # Extract keypoint pixel coordinates and indices for both frames from the feature match
    q_kpt_pixels = np.float32([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_kpt_pixels = np.float32([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # ------------------------------------------------------------------------
    # 2. Compute Essential & Homography matrices
    # ------------------------------------------------------------------------

    # Compute the Essential Matrix
    E, mask_E = cv2.findEssentialMat(q_kpt_pixels, t_kpt_pixels, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    mask_E = mask_E.ravel().astype(bool)

    # Compute the Homography Matrix
    H, mask_H = cv2.findHomography(q_kpt_pixels, t_kpt_pixels, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    mask_H = mask_H.ravel().astype(bool)

    # ------------------------------------------------------------------------
    # 3. Compute symmetric transfer errors & decide which model to use
    # ------------------------------------------------------------------------

    # Compute symmetric transfer error for Essential Matrix
    error_E, num_inliers_E = compute_symmetric_transfer_error(E, q_kpt_pixels, t_kpt_pixels, 'E', K=K)

    # Compute symmetric transfer error for Homography Matrix
    error_H, num_inliers_H = compute_symmetric_transfer_error(H, q_kpt_pixels, t_kpt_pixels, 'H', K=K)

    # Decide which matrix to use based on the ratio of inliers
    if num_inliers_E == 0 and num_inliers_H == 0:
        print("All keypoint pairs yield errors > threshold..")
        return None, False
    ratio = num_inliers_H / (num_inliers_E + num_inliers_H)
    use_homography = (ratio > 0.45)

    # ------------------------------------------------------------------------
    # 4. Recover pose (R, t) from Essential or Homography
    # ------------------------------------------------------------------------

    # Filter keypoints based on the chosen mask
    epipolar_constraint_mask = mask_H if use_homography else mask_E
    matches = matches[epipolar_constraint_mask]
    inlier_q_frame_pixels = q_kpt_pixels[epipolar_constraint_mask]
    inlier_t_frame_pixels = t_kpt_pixels[epipolar_constraint_mask]

    # Check if we will use homography
    R, t, final_match_mask = None, None, None
    if not use_homography:
        # Decompose Essential Matrix
        _, R_est, t_est, mask_pose = cv2.recoverPose(E, inlier_q_frame_pixels, inlier_t_frame_pixels, K)

        # mask_pose indicates inliers used in cv2.recoverPose (1 for inliers, 0 for outliers)
        mask_pose = mask_pose.ravel().astype(bool)

        if R_est is not None and t_est is not None and np.any(mask_pose):
            R, t = R_est, t_est

        # Reprojection filter
        matches = matches[mask_pose]
        reproj_mask = filter_by_reprojection(matches, q_frame, t_frame,R, t, K)
    else:
        # Decompose Homography Matrix
        num_solutions, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)

        # Select the best solution based on criteria
        best_solution = None
        max_front_points = 0
        best_alignment = -1
        desired_normal = np.array([0, 0, 1])

        for i in range(num_solutions):
            R_candidate = Rs[i]
            t_candidate = Ts[i]
            n_candidate = Ns[i]

            # Check if the normal aligns with the 'upward' direction (optional criterion)
            alignment = np.dot(n_candidate, desired_normal)

            # Check if points are in front of camera
            front_points = 0
            invK = np.linalg.inv(K)
            for j in range(len(inlier_q_frame_pixels)):
                # Current frame pixel in camera coords
                p_curr_cam = invK @ np.array([*inlier_q_frame_pixels[j], 1.0])  
                # Previous frame pixel in camera coords
                p_prev_cam = invK @ np.array([*inlier_t_frame_pixels[j], 1.0])

                # Depth for current pixel after transformation
                denom = np.dot(n_candidate, R_candidate @ p_curr_cam + t_candidate)
                depth_curr = np.dot(n_candidate, p_curr_cam) / (denom + 1e-12)  # small eps for safety
                
                # Depth for previous pixel (just dot product since it's reference plane)
                depth_prev = np.dot(n_candidate, p_prev_cam)

                if depth_prev > 0 and depth_curr > 0:
                    front_points += 1

            # Update best solution if it meets criteria
            if front_points > max_front_points and alignment > best_alignment:
                max_front_points = front_points
                best_alignment = alignment
                best_solution = i

        # Use the best solution
        R = Rs[best_solution]
        t = Ts[best_solution]

        # Reprojection filter
        reproj_mask = filter_by_reprojection(q_kpt_pixels, t_kpt_pixels,R, t, K)
    matches = matches[reproj_mask]

    # If we failed to recover R and t
    if R is None or t is None:
        print("Failed to recover a valid pose from either E or H.")
        return None, False

    remaining_points = reproj_mask.sum()
    num_removed_matches = len(matches) - remaining_points

    # ------------------------------------------------------------------------
    # 5. Build the 4x4 Pose matrix
    # ------------------------------------------------------------------------
    # Extract the c1 to c2 pose (this is the transformation that you need to transform a point from the old to the new coordinate system)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.flatten()
    # Extract the c2 to c1 pose (this is the new robot's pose in the old coordinate system)
    inv_pose = invert_transform(pose)

    return inv_pose, True

def filter_by_reprojection(matches, q_frame, t_frame, R, t, K, reproj_threshold=1.0):
    # Extract matched keypoints
    q_pts = np.float32([q_frame.keypoints[m.queryIdx].pt for m in matches])
    t_pts = np.float32([t_frame.keypoints[m.trainIdx].pt for m in matches])

    # Projection matrices
    q_M = K @ np.eye(3,4)        # Reference frame (identity)
    t_M = K @ np.hstack((R, t))  # Current frame

    # Triangulate points
    q_points_4d = cv2.triangulatePoints(q_M, t_M, q_pts.T, t_pts.T)
    q_points_3d = (q_points_4d[:3] / q_points_4d[3]).T

    # Reproject points into the second (current) camera
    t_points_3d = (R @ q_points_3d.T + t).T
    points_proj2, _ = cv2.projectPoints(t_points_3d, np.zeros(3), np.zeros(3), K, None)
    points_proj_px = points_proj2.reshape(-1, 2)

    # Compute reprojection errors
    errors = np.linalg.norm(points_proj_px - t_pts, axis=1)

    # Update the inlier mask
    reproj_mask = errors < reproj_threshold

    num_removed_matches = len(q_pts) - np.sum(reproj_mask)

    return reproj_mask
      
def compute_symmetric_transfer_error(E_or_H, q_kpt_pixels, t_kpt_pixels, matrix_type='E', K=None, threshold=4.0):
    errors = []
    num_inliers = 0

    if matrix_type == 'E':
        F = np.linalg.inv(K.T) @ E_or_H @ np.linalg.inv(K)
    else:
        F = np.linalg.inv(K) @ E_or_H @ K

    # Loop over paired keypoints
    for pt_target, pt_query in zip(t_kpt_pixels, q_kpt_pixels):
        # Convert to homogeneous coordinates
        p1 = np.array([pt_target[0], pt_target[1], 1.0])
        p2 = np.array([pt_query[0], pt_query[1], 1.0])

        # Compute the corresponding epipolar lines
        l2 = F @ p1    # Epipolar line in the query image corresponding to p1
        l1 = F.T @ p2  # Epipolar line in the target image corresponding to p2

        # Normalize the lines using the Euclidean norm of the first two coefficients
        norm_l1 = np.hypot(l1[0], l1[1])
        norm_l2 = np.hypot(l2[0], l2[1])
        if norm_l1 == 0 or norm_l2 == 0:
            continue
        l1_normalized = l1 / norm_l1
        l2_normalized = l2 / norm_l2

        # Compute perpendicular distances (absolute value of point-line dot product)
        d1 = abs(np.dot(p1, l1_normalized))
        d2 = abs(np.dot(p2, l2_normalized))
        error = d1 + d2

        errors.append(error)
        if error < threshold:
            num_inliers += 1

    mean_error = np.mean(errors) if errors else float('inf')
    return mean_error, num_inliers

def is_keyframe(P, t_threshold=3, angle_threshold=20):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    R = P[:3, :3]
    t = P[:3, 3]

    trans = np.sqrt(t[0]**2 + t[2]**2)
    rpy = rotation_matrix_to_euler_angles(R)
    pitch = abs(rpy[1])

    is_keyframe = trans > t_threshold or pitch > angle_threshold

    return is_keyframe


############################### Utility Functions ###############################

def invert_transform(T: np.ndarray) -> np.ndarray:
    # Extract rotation (R) and translation (t)
    R = T[:3, :3]
    t = T[:3, 3]

    # Create an empty 4x4 identity matrix for the result
    T_inv = np.eye(4)

    # R^T goes in the top-left 3x3
    T_inv[:3, :3] = R.T

    # -R^T * t goes in the top-right 3x1
    T_inv[:3, 3] = -R.T @ t

    return T_inv

def transform_points(points_3d: np.ndarray, T: np.ndarray):
    # 1. Convert Nx3 -> Nx4 (homogeneous)
    ones = np.ones((points_3d.shape[0], 1))
    points_hom = np.hstack([points_3d, ones])  # shape (N, 4)

    # 2. Multiply by the transform (assume row vectors)
    transformed_hom = points_hom @ T.T  # shape (N, 4)

    # 3. Normalize back to 3D
    w = transformed_hom[:, 3]
    x = transformed_hom[:, 0] / w
    y = transformed_hom[:, 1] / w
    z = transformed_hom[:, 2] / w
    transformed_3d = np.column_stack((x, y, z))

    return transformed_3d

def isnan(p: np.ndarray):
    if np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]):
        return True
    return False

def delete_subdirectories(data_dir):
    # Convert to Path object if not already
    data_dir = Path(data_dir)
    if not os.path.isdir(data_dir):
        return

    # Iterate through the contents of the directory
    for item in data_dir.iterdir():
        # Check if the item is a directory
        if item.is_dir():
            # Recursively delete the directory and its contents
            shutil.rmtree(item)

def save_2_images(image1, image2, save_path):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert images to NumPy arrays if they are PIL images
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)

    # Ensure both images are in a format that can be saved by OpenCV
    if isinstance(image1, np.ndarray):
        if len(image1.shape) == 3 and image1.shape[2] == 3:  # Check if it's a 3-channel image
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    if isinstance(image2, np.ndarray):
        if len(image2.shape) == 3 and image2.shape[2] == 3:  # Check if it's a 3-channel image
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    # Ensure both images have the same height
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    if height1 != height2:
        # Resize the second image to match the height of the first image
        image2 = cv2.resize(image2, (int(width2 * height1 / height2), height1))

    # Concatenate images side by side
    combined_image = np.hstack((image1, image2))

    # Save the combined image using OpenCV
    cv2.imwrite(save_path, combined_image)

def save_depth(image, save_path):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the image is a PIL image, convert it to a NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # If the image is a NumPy array, ensure it's in a format that can be saved by OpenCV
    if isinstance(image, np.ndarray):
        # Convert the image to BGR format if it's in RGB (PIL is usually in RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:  # Check if it's a 3-channel image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save the image using OpenCV
    depth_path = save_path.parent / (save_path.name + '_depth.png')
    cv2.imwrite(depth_path, image)

    # Convert the depth image to 3d points
    points = depth_to_3d_points(image)

    # Extract the Z (depth) component from the points
    Z = points[:, 2]
    
    # Normalize the Z values to an 8-bit range
    # Handling NaNs or Infs if they exist:
    Z = Z[np.isfinite(Z)]
    if Z.size == 0:
        print("Warning: No valid Z points to create a heatmap.")
        return
    
    # Z_norm is a 1D array. Reshape into a w*h image
    h, w = image.shape[:2]
    Z_img = Z.reshape(h, w)
    
    # Use matplotlib to create a heatmap with a colorbar
    plt.figure(figsize=(8, 6))
    plt.imshow(Z_img, cmap='jet', aspect='auto', origin='upper')
    plt.colorbar(label="Depth (m)")
    plt.title("Depth Heatmap")
    heatmap_path = save_path.parent / (save_path.name + '_heat.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    
def save_image(image, save_path):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the image is a PIL image, convert it to a NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # If the image is a NumPy array, ensure it's in a format that can be saved by OpenCV
    if isinstance(image, np.ndarray):
        # Convert the image to BGR format if it's in RGB (PIL is usually in RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:  # Check if it's a 3-channel image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save the image using OpenCV
    cv2.imwrite(save_path, image)

def rotation_matrix_to_euler_angles(R):
    """Converts a Rotation Matrix to roll, pitch, yaw euler angles"""
    
    sy = np.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def depth_to_3d_points(depth_image, cx = 319.5, cy = 239.5, fx=525.0, fy=525.0, factor=5000):
    height, width = depth_image.shape
    points = []

    for v in range(height):
        for u in range(width):
            Z = depth_image[v, u] / factor
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append((X, Y, Z))

    return np.array(points, dtype=np.float64)

def keypoints_depth_to_3d_points(kpts, depth_image, cx, cy, fx, fy, factor=5000):
    """
    Convert 2D keypoints to 3D points using the depth image and camera intrinsics.

    Parameters:
        kpts (np.ndarray): The 2D keypoints (Nx2).
        depth_image (np.ndarray): The depth image.
        cx (float): The x-coordinate of the principal point.
        cy (float): The y-coordinate of the principal point.
        fx (float): The focal length in the x-axis.
        fy (float): The focal length in the y-axis.
        factor (float): The scaling factor for depth values (default is 5000 for 16-bit depth images).

    Returns:
        np.ndarray: An array of 3D points.
    """
    points_3d = []
    valid_points_indices = []
    j = 0
    for i, pt in enumerate(kpts):
        # Extract pixel coordinates
        u, v = int(pt[0]), int(pt[1])

        # Extract the depth (z) at that coordinate
        Z = depth_image[v, u] / factor

        # Skip points with zero depth and too far away points
        if Z <= 0:
            j += 1
            continue
        # if Z > 5:
        #     j += 1
        #     continue

        # Convert the pixel coordinates to the X, Y coordinates
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points_3d.append((X, Y, Z))
        valid_points_indices.append(i)
    # print(f"Removed {j}/{len(kpts)} points that are too far away!")

    points_3d = np.array(points_3d, dtype=np.float64)
    valid_points_indices = np.array(valid_points_indices, dtype=np.uint64)
    
    return points_3d, valid_points_indices


############################### Visualization Functions ###############################

def init_traj_plot_3d():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'b-', label='Trajectory')

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")
    ax.set_title("Trajectory (Real-Time)")
    ax.legend()

    fig.show()
    return fig, ax, line


def update_traj_plot_3d(fig, ax, line, poses):
    poses = np.array(poses)
    if poses.shape[0] < 2:
        return

    x = poses[:, 0, 3]
    z = poses[:, 2, 3]
    ny = -poses[:, 1, 3]

    line.set_data(x, z)
    line.set_3d_properties(ny)

    # keep axes sane (simple autoscale)
    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)
