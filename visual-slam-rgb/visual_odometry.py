import cv2
import numpy as np
from scipy.signal import savgol_filter

def epipolar_error_stats(F, pts1, pts2):
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)

    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}

    x1 = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    x2 = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    l2 = (F @ x1.T).T
    l1 = (F.T @ x2.T).T

    d2 = np.abs(np.sum(l2 * x2, axis=1)) / (
        np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2) + 1e-12
    )
    d1 = np.abs(np.sum(l1 * x1, axis=1)) / (
        np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2) + 1e-12
    )

    e = 0.5 * (d1 + d2)
    
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}


    return {
        "n": int(e.shape[0]),
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "p95": float(np.percentile(e, 95)),
    }

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
            return None, None, None

        matches = self.matcher.match(f1.descriptors, f2.descriptors)  # matching

        # check if there enough matches
        if len(matches) < 8:
            return None, None, None

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

        #epipolar error
        Kinv = np.linalg.inv(self.k)
        F = Kinv.T @ essential_matrix @ Kinv
        inliers_idx = np.where(inliers)[0]

        info = {
            "matches_raw": matches[:50],
            "matches_inliers": [matches[i] for i in inliers_idx[:50]],
            "epi_raw": epipolar_error_stats(F, pts1, pts2),
            "epi_in": epipolar_error_stats(F, pts1_in, pts2_in),
        }

        if pts1_in.shape[0] < 8:
            return None, None, info

        _, rotation, translation, _ = cv2.recoverPose(
            essential_matrix, pts1_in, pts2_in, self.k
        )

        return rotation, translation, info

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
        min_inliers = 20
        max_inlier_median_epipolar_px = 2.0
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
                cv2.waitKey(1)
                continue

            prev = self.frames[i - 1]
            rotation, translation, info = self.estimate_motion(prev, frame)

            # matches windows + epipolar error stats
            if info is not None:
                cv2.imshow(
                    "Matches (Raw)",
                    cv2.drawMatches(
                        prev.image, prev.keypoints,
                        frame.image, frame.keypoints,
                        info["matches_raw"], None
                    )
                )

                cv2.imshow(
                    "Matches (Inliers)",
                    cv2.drawMatches(
                        prev.image, prev.keypoints,
                        frame.image, frame.keypoints,
                        info["matches_inliers"], None
                    )
                )

                r = info["epi_raw"]
                i_ = info["epi_in"]
                print(
                    f"[Frame {i}] Epipolar error (px) "
                    f"Before Filtering: n={r['n']} mean={r['mean']:.2f} med={r['median']:.2f} p95={r['p95']:.2f} | "
                    f"After Filtering: n={i_['n']} mean={i_['mean']:.2f} med={i_['median']:.2f} p95={i_['p95']:.2f}"
                )

            if rotation is None or translation is None:
                print(f"[Frame {i}] Rejected: pose estimation failed")
                cv2.waitKey(1)
                continue  # if pose estimation failed, the frame is skipped

            #trajectory reliability check
            if info is None:
                print(f"[Frame {i}] Rejected: no info")
                cv2.waitKey(1)
                continue

            if info["epi_in"]["n"] < min_inliers:
                print(f"[Frame {i}] Rejected: too few inliers ({info['epi_in']['n']})")
                cv2.waitKey(1)
                continue

            if info["epi_in"]["median"] > max_inlier_median_epipolar_px:
                print(
                    f"[Frame {i}] Rejected: high inlier epipolar median "
                    f"({info['epi_in']['median']:.2f}px)"
                )
                cv2.waitKey(1)
                continue

            frame.rotation_matrix = rotation
            frame.translation_vector = translation

            t_matrix = np.eye(4)
            t_matrix[:3, :3] = frame.rotation_matrix
            t_matrix[:3, 3] = frame.translation_vector.flatten()

            frame.pose = prev.pose @ np.linalg.inv(t_matrix)

            pos = frame.pose[:3, 3]
            self.trajectory.append(pos)

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
