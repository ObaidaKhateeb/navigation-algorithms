import cv2
import numpy as np
from scipy.signal import savgol_filter


class Frame:
    def __init__(self, img, idx):
        self.id = idx
        self.image = img
        self.keypoints = None
        self.descriptors = None
        self.pose = np.eye(4)
        self.rotation_matrix = np.eye(3)
        self.translation_vector = np.zeros((3, 1))
        self.processed = False


class VisualOdometry:
    def __init__(self, image_paths, visualizer, use_sift=False, smooth=False):
        self.images = image_paths
        self.use_sift = use_sift
        self.smooth = smooth
        self.orb = cv2.ORB_create(2000)
        self.sift = cv2.SIFT_create()
        self.visualizer = visualizer
        if self.use_sift:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.frames = []
        self.trajectory = []
        self.k = None

    def extract_features(self, frame):
        detector = self.sift if self.use_sift else self.orb
        kp, des = detector.detectAndCompute(frame.image, None)
        frame.keypoints = kp
        frame.descriptors = des

    def estimate_motion(self, f1, f2):
        # ensure descriptors exists
        if f1.descriptors is None or f2.descriptors is None:
            return None, None, []

        matches = self.matcher.match(f1.descriptors, f2.descriptors)  # matching

        # check if there enough matches
        if len(matches) < 8:
            return None, None, []

        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:2000]

        # point arrays building
        pts1 = np.float32(
            [f1.keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 2)
        pts2 = np.float32(
            [f2.keypoints[m.trainIdx].pt for m in matches]
        ).reshape(-1, 2)

        # Essential matrix computation
        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            return None, None, matches
        essential_matrix, mask = cv2.findEssentialMat(
            pts1, pts2, self.k,
            method=cv2.LMEDS, prob=0.999, threshold=1.0
        )
        if essential_matrix is None or mask is None:
            return None, None, matches

        # Pose recovering
        inliers = mask.ravel().astype(bool)
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        if pts1_in.shape[0] < 8:
            return None, None, matches

        _, rotation, translation, _ = cv2.recoverPose(
            essential_matrix, pts1_in, pts2_in, self.k
        )

        return rotation, translation, matches

    def smooth_traj(self, traj):
        try:
            if traj.shape[0] < 12:  # smaller than the window size
                return traj

            sm = traj.copy()
            for d in range(3):
                sm[1:, d] = savgol_filter(
                    traj[1:, d], window_length=11,
                    polyorder=3, mode="interp"
                )

            return sm
        except Exception:
            return traj

    def run(self):
        total_frames = len(self.images)
        for i, path in enumerate(self.images):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            frame = Frame(img, i)

            if self.k is None:
                h, w = img.shape
                fx = fy = 0.8 * w
                cx = w / 2
                cy = h / 2
                self.k = np.array(
                    [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ]
                )

            self.extract_features(frame)
            self.frames.append(frame)

            kp_img = cv2.drawKeypoints(
                frame.image, frame.keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.imshow("Features (Current Frame)", kp_img)

            if i == 0:
                self.trajectory.append(np.array([0.0, 0.0, 0.0]))
                traj = np.array(self.trajectory)
                if self.smooth:
                    traj = self.smooth_traj(traj)
                self.visualizer.update(traj, i, total_frames)
                continue

            prev = self.frames[i - 1]
            rotation, translation, matches = self.estimate_motion(prev, frame)

            if rotation is None or translation is None:
                continue  # if pose estimation failed, the frame is skipped

            frame.rotation_matrix = rotation
            frame.translation_vector = translation

            t_matrix = np.eye(4)
            t_matrix[:3, :3] = frame.rotation_matrix
            t_matrix[:3, 3] = frame.translation_vector.flatten()

            frame.pose = prev.pose @ np.linalg.inv(t_matrix)

            pos = frame.pose[:3, 3]
            self.trajectory.append(pos)

            # keypoints/matches window
            vis = cv2.drawMatches(
                prev.image, prev.keypoints,
                frame.image, frame.keypoints,
                matches[:50], None
            )
            cv2.imshow("Top Matches", vis)

            # trajectory window
            traj = np.array(self.trajectory)
            if self.smooth:
                traj = self.smooth_traj(traj)
            self.visualizer.update(traj, i, total_frames)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                raise SystemExit

        traj = np.array(self.trajectory)
        if self.smooth:
            traj = self.smooth_traj(traj)
        frame_count = len(self.images)
        self.visualizer.update(traj, frame_count, total_frames)
        while True:
            self.visualizer.update(
                traj, frame_count, total_frames
            )  # keep trajectory window responsive

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

        cv2.destroyAllWindows()
