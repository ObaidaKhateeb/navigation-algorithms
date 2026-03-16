import cv2
import numpy as np
from scipy.signal import savgol_filter
#from scipy.optimize import least_squares

class Point3D:
    def __init__(self, xyz, pid):
        self.id = pid
        self.xyz = np.asarray(xyz, dtype=float).reshape(3,)
        self.observations = []  # list of (frame_id, kp_idx)


class Map3D:
    def __init__(self):
        self.frames = []
        self.points = []   # list of Point3D
        self._next_pid = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_points(self, X_world: np.ndarray):
        """
        X_world: Nx3 array of 3D points in WORLD coordinates
        """
        if X_world is None:
            return
        X_world = np.asarray(X_world, dtype=float).reshape(-1, 3)
        if X_world.shape[0] == 0:
            return

        for i in range(X_world.shape[0]):
            self.points.append(Point3D(X_world[i], self._next_pid))
            self._next_pid += 1

    def points_array(self):
        if len(self.points) == 0:
            return np.zeros((0, 3), dtype=float)
        return np.stack([p.xyz for p in self.points], axis=0)


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
        self.pose = np.eye(4)  # we treat as Twc (camera -> world)
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
        

        self.map = Map3D()
        # ---------- Part 5: PnP database ----------
        self.map_db_xyz = np.zeros((0, 3), dtype=np.float32)
        self.map_db_des = None
        # ------------------------------------------

    def extract_features(self, frame: Frame):
        detector = self.sift if self.use_sift else self.orb
        kp, des = detector.detectAndCompute(frame.image, None)
        frame.keypoints = kp
        frame.descriptors = des

    def estimate_motion(self, f1: Frame, f2: Frame):
        if f1.descriptors is None or f2.descriptors is None:
            return None, None, None

        matches = self.matcher.match(f1.descriptors, f2.descriptors)
        if len(matches) < 8:
            return None, None, None

        matches = sorted(matches, key=lambda x: x.distance)[:2000]

        pts1 = np.float32([f1.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([f2.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            return None, None, None

        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.k,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            return None, None, None

        inliers = mask.ravel().astype(bool)
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]
        inliers_idx = np.where(inliers)[0]

        # Epipolar error stats (via F = K^-T E K^-1)
        Kinv = np.linalg.inv(self.k)
        F = Kinv.T @ E @ Kinv

        info = {
            # display
            "matches_raw": matches[:50],
            "matches_inliers": [matches[i] for i in inliers_idx[:50]],

            # full inliers for triangulation
            "inliers_idx": inliers_idx,
            "matches_inliers_all": [matches[i] for i in inliers_idx],
            "pts1_in": pts1_in,
            "pts2_in": pts2_in,

            # stats
            "epi_raw": epipolar_error_stats(F, pts1, pts2),
            "epi_in": epipolar_error_stats(F, pts1_in, pts2_in),
        }

        if pts1_in.shape[0] < 8:
            return None, None, info

        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.k)
        return R, t, info

    def smooth_traj(self, traj: np.ndarray) -> np.ndarray:
        try:
            if traj.shape[0] < 12:
                return traj
            sm = traj.copy()
            for d in range(3):
                sm[1:, d] = savgol_filter(
                    traj[1:, d], window_length=11, polyorder=3, mode="interp"
                )
            return sm
        except Exception:
            return traj

    # ---------- Part 4 helpers (must be INSIDE the class) ----------
    def _Tcw_from_Twc(self, Twc: np.ndarray) -> np.ndarray:
        return np.linalg.inv(Twc)

    def _build_projection(self, Twc: np.ndarray) -> np.ndarray:
        """
        Build P = K [R|t] using world->camera transform.
        Twc is camera->world.
        """
        Tcw = self._Tcw_from_Twc(Twc)
        return self.k @ Tcw[:3, :]  # 3x4

    def _triangulate_and_filter(self, Twc1, Twc2, pts1, pts2):
        pts1 = np.asarray(pts1, dtype=float).reshape(-1, 2)
        pts2 = np.asarray(pts2, dtype=float).reshape(-1, 2)
        if pts1.shape[0] < 8:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=bool)

        P1 = self._build_projection(Twc1)
        P2 = self._build_projection(Twc2)

        X_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # 4xN
        X = (X_h[:3, :] / (X_h[3, :] + 1e-12)).T            # Nx3

        good = np.all(np.isfinite(X), axis=1)

        Tcw1 = self._Tcw_from_Twc(Twc1)
        Tcw2 = self._Tcw_from_Twc(Twc2)

        X1 = (Tcw1[:3, :3] @ X.T + Tcw1[:3, 3:4]).T
        X2 = (Tcw2[:3, :3] @ X.T + Tcw2[:3, 3:4]).T

        good &= (X1[:, 2] > 0.0) & (X2[:, 2] > 0.0)

        max_depth = 20.0
        good &= (np.abs(X1[:, 2]) < max_depth) & (np.abs(X2[:, 2]) < max_depth)

        return X[good], good
    
    def _project_point(self, Twc, Xw):
        """
        Project one 3D world point to image pixels using Twc (camera->world).
        Returns (u, v) or None if point is behind camera.
        """
        Tcw = self._Tcw_from_Twc(Twc)
        Xw = np.asarray(Xw, dtype=float).reshape(3, 1)

        Xc = Tcw[:3, :3] @ Xw + Tcw[:3, 3:4]
        z = Xc[2, 0]
        if z <= 1e-6:
            return None

        x = self.k @ Xc
        u = x[0, 0] / x[2, 0]
        v = x[1, 0] / x[2, 0]
        return np.array([u, v], dtype=float)
    
    def _pose_to_params_cw(self, Twc):
        """
        Convert pose from Twc (camera->world) to 6D params of Tcw (world->camera):
        [rx, ry, rz, tx, ty, tz]
        where rotation is Rodrigues vector.
        """
        Tcw = np.linalg.inv(Twc)
        Rcw = Tcw[:3, :3]
        tcw = Tcw[:3, 3]
        rvec, _ = cv2.Rodrigues(Rcw)
        return np.hstack([rvec.ravel(), tcw.ravel()])


    def _params_to_pose_wc(self, params):
        """
        Convert 6D params [rx, ry, rz, tx, ty, tz] of Tcw (world->camera)
        back to Twc (camera->world).
        """
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)

        Rcw, _ = cv2.Rodrigues(rvec)

        Tcw = np.eye(4)
        Tcw[:3, :3] = Rcw
        Tcw[:3, 3] = tvec.ravel()

        Twc = np.linalg.inv(Tcw)
        return Twc

    def _check_motion_validity(self, prev_pose, new_pose, max_translation=3.0, max_rotation_deg=20.0):
        """
        Check if the motion between two poses is reasonable.
        Returns (is_valid, reason_string).
        """
        # Compute relative transformation
        T_rel = np.linalg.inv(prev_pose) @ new_pose
        
        # Check translation magnitude
        translation = T_rel[:3, 3]
        trans_magnitude = np.linalg.norm(translation)
        
        if trans_magnitude > max_translation:
            return False, f"translation too large: {trans_magnitude:.2f}"
        
        # Check rotation magnitude
        R = T_rel[:3, :3]
        # Rotation angle from rotation matrix: angle = arccos((trace(R) - 1) / 2)
        trace = np.trace(R)
        # Clamp to avoid numerical issues with arccos
        trace_clamped = np.clip((trace - 1) / 2, -1.0, 1.0)
        angle_rad = np.arccos(trace_clamped)
        angle_deg = np.degrees(angle_rad)
        
        if angle_deg > max_rotation_deg:
            return False, f"rotation too large: {angle_deg:.1f}°"
        
        return True, "OK"
    
    def _frame_reprojection_residuals(self, params, frame_obs):
        """
        params: 6D pose params [rvec(3), tvec(3)] in world->camera form
        frame_obs: list of (Xw, uv_obs)
        returns residual vector [du1, dv1, du2, dv2, ...]
        """
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)
        Rcw, _ = cv2.Rodrigues(rvec)

        residuals = []

        for Xw, uv_obs in frame_obs:
            Xw = np.asarray(Xw, dtype=float).reshape(3, 1)
            Xc = Rcw @ Xw + tvec
            z = Xc[2, 0]

            if z <= 1e-6:
                continue

            x = self.k @ Xc
            u = x[0, 0] / x[2, 0]
            v = x[1, 0] / x[2, 0]

            residuals.extend([u - uv_obs[0], v - uv_obs[1]])

        return np.array(residuals, dtype=float)
    
    def _frame_reprojection_cost(self, params, frame_obs, robust_clip=25.0):
        """
        Sum of squared reprojection residuals with optional clipping.
        robust_clip=25 means each residual component squared is clipped at 25.
        """
        res = self._frame_reprojection_residuals(params, frame_obs)
        if res.size == 0:
            return 0.0

        sq = res ** 2
        sq = np.minimum(sq, robust_clip)
        return float(np.sum(sq))
    
    def _numerical_gradient_pose(self, params, frame_obs, eps=1e-5):
        """
        Numerical gradient of reprojection cost wrt 6 pose params.
        """
        grad = np.zeros_like(params)

        for k in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()

            p_plus[k] += eps
            p_minus[k] -= eps

            c_plus = self._frame_reprojection_cost(p_plus, frame_obs)
            c_minus = self._frame_reprojection_cost(p_minus, frame_obs)

            grad[k] = (c_plus - c_minus) / (2.0 * eps)

        return grad
    
    def detect_loop_closure(self, current_frame_id, min_frame_gap=100, min_matches=100):
        """
        Detect if current frame matches a previously visited location.
        Returns: (is_loop, matched_frame_id) or (False, None)
        """
        if current_frame_id < min_frame_gap:
            return False, None
        
        current_frame = self.frames[current_frame_id]
        if current_frame.descriptors is None or len(current_frame.descriptors) < 50:
            return False, None
        
        # Search for matches with older frames (skip recent ones to avoid matching nearby frames)
        best_match_count = 0
        best_match_frame = None
        
        for old_frame_id in range(0, current_frame_id - min_frame_gap, 5):
            old_frame = self.frames[old_frame_id]
            if old_frame.descriptors is None:
                continue
            
            # Match descriptors
            try:
                matches = self.matcher.match(current_frame.descriptors, old_frame.descriptors)
            except:
                continue
            
            if len(matches) < min_matches:
                continue
            
            # Sort by quality and count good matches
            good_matches = sorted(matches, key=lambda x: x.distance)[:150]
            
            # Basic descriptor quality check
            avg_distance = np.mean([m.distance for m in good_matches])
            threshold = 40 if self.use_sift else 25

            if avg_distance >= threshold:
                continue

            # Geometric verification with Essential matrix
            pts_cur = np.float32([
                current_frame.keypoints[m.queryIdx].pt for m in good_matches
            ]).reshape(-1, 2)

            pts_old = np.float32([
                old_frame.keypoints[m.trainIdx].pt for m in good_matches
            ]).reshape(-1, 2)

            E, mask = cv2.findEssentialMat(
                pts_old, pts_cur, self.k,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )

            if E is None or mask is None:
                continue

            geom_inliers = int(mask.ravel().sum())

            if geom_inliers > best_match_count:
                best_match_count = geom_inliers
                best_match_frame = old_frame_id
        
        # Found a good loop closure candidate?
        if best_match_count >= min_matches:
            print(f"\n{'='*60}")
            print(f"[LOOP CLOSURE] Frame {current_frame_id} matches Frame {best_match_frame}")
            print(f"[LOOP CLOSURE] Geometric inliers: {best_match_count}")            
            print(f"{'='*60}\n")
            return True, best_match_frame
        
        return False, None
    
    
    def reprojection_error_stats(self):
        errs = []

        for p in self.map.points:
            Xw = p.xyz

            for frame_id, kp_idx in p.observations:
                if frame_id < 0 or frame_id >= len(self.frames):
                    continue

                fr = self.frames[frame_id]
                if fr.keypoints is None or kp_idx < 0 or kp_idx >= len(fr.keypoints):
                    continue

                uv_obs = np.array(fr.keypoints[kp_idx].pt, dtype=float)
                uv_proj = self._project_point(fr.pose, Xw)

                if uv_proj is None:
                    continue

                e = np.linalg.norm(uv_proj - uv_obs)
                if np.isfinite(e):
                    errs.append(e)

        if len(errs) == 0:
            return {"n": 0, "mean": 0.0, "median": 0.0, "rmse": 0.0, "p95": 0.0}

        errs = np.asarray(errs, dtype=float)
        return {
            "n": int(errs.size),
            "mean": float(np.mean(errs)),
            "median": float(np.median(errs)),
            "rmse": float(np.sqrt(np.mean(errs ** 2))),
            "p95": float(np.percentile(errs, 95)),
        }
        
        
    def optimize_poses(self, window_size=20):
        """
        Part 7: Gradient Descent pose-only optimization.
        Optimizes rotation + translation (6 DoF) for recent frames
        by minimizing reprojection error.
        """
        if len(self.map.points) == 0 or len(self.frames) < 2:
            return

        total_frames = len(self.frames)

        if total_frames <= window_size:
            start_idx = 1   # keep frame 0 fixed
            frames_to_optimize = self.frames[1:]
        else:
            start_idx = total_frames - window_size
            frames_to_optimize = self.frames[start_idx:]

        # Collect observations per frame
        observations_by_frame = {}
        for fr in frames_to_optimize:
            observations_by_frame[fr.id] = []

        for p in self.map.points:
            for fid, kp_idx in p.observations:
                if fid < start_idx or fid >= total_frames:
                    continue

                fr = self.frames[fid]
                if fr.keypoints is None or kp_idx < 0 or kp_idx >= len(fr.keypoints):
                    continue

                uv_obs = np.array(fr.keypoints[kp_idx].pt, dtype=float)
                observations_by_frame[fid].append((p.xyz.copy(), uv_obs))

        total_obs = sum(len(v) for v in observations_by_frame.values())
        if total_obs < 20:
            print(f"  [GD] Skipping: only {total_obs} observations")
            return

        print(f"  [GD] Optimizing {len(frames_to_optimize)} frames with {total_obs} observations")

        # Hyperparameters
        max_iterations = 20
        base_lr = 1e-4
        min_step_norm = 1e-6
        min_improvement = 1e-4

        total_cost_before = 0.0
        for fr in frames_to_optimize:
            frame_obs = observations_by_frame[fr.id]
            if len(frame_obs) < 10:
                continue
            params = self._pose_to_params_cw(fr.pose)
            total_cost_before += self._frame_reprojection_cost(params, frame_obs)

        # Optimize each frame independently (pose-only refinement)
        for fr in frames_to_optimize:
            frame_obs = observations_by_frame[fr.id]

            if len(frame_obs) < 10:
                continue

            params = self._pose_to_params_cw(fr.pose)
            lr = base_lr
            prev_cost = self._frame_reprojection_cost(params, frame_obs)

            for _ in range(max_iterations):
                grad = self._numerical_gradient_pose(params, frame_obs, eps=1e-5)

                # Gradient clipping
                grad[:3] = np.clip(grad[:3], -100.0, 100.0)
                grad[3:] = np.clip(grad[3:], -100.0, 100.0)

                step = lr * grad

                # Step clipping
                step[:3] = np.clip(step[:3], -1e-2, 1e-2)
                step[3:] = np.clip(step[3:], -1e-1, 1e-1)

                new_params = params - step
                new_cost = self._frame_reprojection_cost(new_params, frame_obs)

                if new_cost < prev_cost:
                    improvement = prev_cost - new_cost
                    params = new_params
                    prev_cost = new_cost

                    if improvement < min_improvement or np.linalg.norm(step) < min_step_norm:
                        break
                else:
                    lr *= 0.5
                    if lr < 1e-7:
                        break

            fr.pose = self._params_to_pose_wc(params)

        total_cost_after = 0.0
        for fr in frames_to_optimize:
            frame_obs = observations_by_frame[fr.id]
            if len(frame_obs) < 10:
                continue
            params = self._pose_to_params_cw(fr.pose)
            total_cost_after += self._frame_reprojection_cost(params, frame_obs)

        print(f"  [GD] Cost before: {total_cost_before:.2f}")
        print(f"  [GD] Cost after : {total_cost_after:.2f}")



    def correct_loop_closure(self, current_frame_id, loop_frame_id, window_size=150):
        """
        Gradient Descent pose-only loop closure optimization.
        Optimizes rotation + translation for a local window of frames,
        while encouraging the current frame to align with the matched loop frame.
        """
        total_frames = len(self.frames)
        if total_frames < 2:
            return

        start_idx = max(loop_frame_id + 1, current_frame_id - window_size + 1)
        end_idx = current_frame_id + 1
        frames_to_optimize = self.frames[start_idx:end_idx]

        if len(frames_to_optimize) == 0:
            print("  [GD-Loop] Skipping: no frames")
            return

        observations_by_frame = {fr.id: [] for fr in frames_to_optimize}

        for p in self.map.points:
            for fid, kp_idx in p.observations:
                if fid < start_idx or fid >= end_idx:
                    continue

                fr = self.frames[fid]
                if fr.keypoints is None or kp_idx < 0 or kp_idx >= len(fr.keypoints):
                    continue

                uv = np.array(fr.keypoints[kp_idx].pt, dtype=float)
                observations_by_frame[fid].append((p.xyz.copy(), uv))

        total_obs = sum(len(v) for v in observations_by_frame.values())
        if total_obs < 20:
            print("  [GD-Loop] Not enough observations")
            return

        print(f"  [GD-Loop] Optimizing {len(frames_to_optimize)} frames")

        target_params = self._pose_to_params_cw(self.frames[loop_frame_id].pose)

        max_iterations = 15
        base_lr = 5e-4
        rot_weight = 500.0
        trans_weight = 500.0
        min_improvement = 1e-4
        min_step_norm = 1e-6

        def loop_cost(params, frame_id):
            if frame_id != current_frame_id:
                return 0.0

            r = params[:3] - target_params[:3]
            t = params[3:] - target_params[3:]

            return rot_weight * np.sum(r * r) + trans_weight * np.sum(t * t)

        def total_cost(params, obs, frame_id):
            return self._frame_reprojection_cost(params, obs) + loop_cost(params, frame_id)

        def numerical_grad(params, obs, frame_id):
            eps = 1e-5
            grad = np.zeros_like(params)

            for i in range(len(params)):
                p1 = params.copy()
                p2 = params.copy()

                p1[i] += eps
                p2[i] -= eps

                c1 = total_cost(p1, obs, frame_id)
                c2 = total_cost(p2, obs, frame_id)

                grad[i] = (c1 - c2) / (2 * eps)

            return grad

        total_cost_before = 0.0
        for fr in frames_to_optimize:
            obs = observations_by_frame[fr.id]
            if len(obs) < 10:
                continue
            params = self._pose_to_params_cw(fr.pose)
            total_cost_before += total_cost(params, obs, fr.id)

        changed_any = False

        for fr in frames_to_optimize:
            obs = observations_by_frame[fr.id]

            if len(obs) < 10:
                continue

            params = self._pose_to_params_cw(fr.pose)
            lr = base_lr
            prev_cost = total_cost(params, obs, fr.id)

            for _ in range(max_iterations):
                grad = numerical_grad(params, obs, fr.id)

                # Gradient clipping
                grad[:3] = np.clip(grad[:3], -200.0, 200.0)
                grad[3:] = np.clip(grad[3:], -200.0, 200.0)

                step = lr * grad

                # Step clipping
                step[:3] = np.clip(step[:3], -5e-2, 5e-2)
                step[3:] = np.clip(step[3:], -5e-1, 5e-1)

                new_params = params - step
                new_cost = total_cost(new_params, obs, fr.id)

                if new_cost < prev_cost:
                    improvement = prev_cost - new_cost
                    params = new_params
                    prev_cost = new_cost
                    changed_any = True

                    if improvement < min_improvement or np.linalg.norm(step) < min_step_norm:
                        break
                else:
                    lr *= 0.5
                    if lr < 1e-7:
                        break

            fr.pose = self._params_to_pose_wc(params)

        total_cost_after = 0.0
        for fr in frames_to_optimize:
            obs = observations_by_frame[fr.id]
            if len(obs) < 10:
                continue
            params = self._pose_to_params_cw(fr.pose)
            total_cost_after += total_cost(params, obs, fr.id)

        print(f"  [GD-Loop] Cost before: {total_cost_before:.2f}")
        print(f"  [GD-Loop] Cost after : {total_cost_after:.2f}")

        if not changed_any:
            print("  [GD-Loop] Warning: loop optimization made almost no pose updates")
     
    # ---------- Main loop ----------
    def run(self):
        min_inliers = 20
        max_inlier_median_epipolar_px = 2.0
        total_frames = len(self.images)

        # ---------- Part 5 params ----------
        pnp_interval = 5              # every N frames do PnP relocalization
        min_pnp_corr = 30             # minimum 2D-3D matches to attempt PnP
        pnp_reproj_err = 4.0          # RANSAC reprojection threshold (pixels)
        ratio = 0.75                  # Lowe ratio
        # ----------------------------------

        # Create a matcher for PnP DB matching (query=current frame, train=map DB)
        if self.use_sift:
            pnp_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            pnp_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        for i, path in enumerate(self.images):
            did_optimize = False
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[Frame {i}] Failed to read image: {path}")
                continue

            frame = Frame(img, i)

            if self.k is None:
                h, w = img.shape
                fx = fy = 0.8 * w
                cx = w / 2
                cy = h / 2
                self.k = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=float)

            self.extract_features(frame)
            self.frames.append(frame)
            self.map.add_frame(frame)

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
                pts = self.map.points_array()
                self.visualizer.update(traj, pts, i, total_frames)
                cv2.waitKey(1)
                continue

            prev = self.frames[i - 1]
            R, t, info = self.estimate_motion(prev, frame)

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
                    f"Before: n={r['n']} mean={r['mean']:.2f} med={r['median']:.2f} p95={r['p95']:.2f} | "
                    f"After:  n={i_['n']} mean={i_['mean']:.2f} med={i_['median']:.2f} p95={i_['p95']:.2f}"
                )

            if R is None or t is None or info is None:
                print(f"[Frame {i}] Rejected: pose estimation failed")
                # Keep previous pose to maintain trajectory continuity
                if len(self.frames) > 1:
                    frame.pose = self.frames[-2].pose.copy()
                    self.trajectory.append(frame.pose[:3, 3].copy())
                cv2.waitKey(1)
                continue

            if info["epi_in"]["n"] < min_inliers:
                print(f"[Frame {i}] Rejected: too few inliers ({info['epi_in']['n']})")
                # Keep previous pose to maintain trajectory continuity
                if len(self.frames) > 1:
                    frame.pose = self.frames[-2].pose.copy()
                    self.trajectory.append(frame.pose[:3, 3].copy())
                cv2.waitKey(1)
                continue

            if info["epi_in"]["median"] > max_inlier_median_epipolar_px:
                print(f"[Frame {i}] Rejected: high inlier epipolar median ({info['epi_in']['median']:.2f}px)")
                # Keep previous pose to maintain trajectory continuity
                if len(self.frames) > 1:
                    frame.pose = self.frames[-2].pose.copy()
                    self.trajectory.append(frame.pose[:3, 3].copy())
                cv2.waitKey(1)
                continue

            frame.rotation_matrix = R
            frame.translation_vector = t

            # Relative transform (camera_i -> camera_{i+1}) from recoverPose
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = t.flatten()

            # Compute new pose
            new_pose = prev.pose @ np.linalg.inv(T_rel)

            # CHECK MOTION VALIDITY
            is_valid, reason = self._check_motion_validity(prev.pose, new_pose, max_translation=3.0, max_rotation_deg=45.0)
            if not is_valid:
                print(f"[Frame {i}] Rejected: unrealistic motion ({reason})")
                # Keep previous pose to maintain trajectory continuity
                if len(self.frames) > 1:
                    frame.pose = self.frames[-2].pose.copy()
                    self.trajectory.append(frame.pose[:3, 3].copy())
                cv2.waitKey(1)
                continue

            # Motion is valid - accept new pose
            # Additional safety: reject if position is unreasonable
            if np.linalg.norm(new_pose[:3, 3]) < 1000.0:
                frame.pose = new_pose
            else:
                # Keep previous pose if new one is extreme
                if len(self.frames) > 1:
                    frame.pose = self.frames[-2].pose.copy()
                print(f"[Frame {i}] Rejected: position too extreme ({np.linalg.norm(new_pose[:3, 3]):.2f})")

            # ---------------- Part 4: triangulate + add to map ----------------
            pts1_in = info.get("pts1_in", None)
            pts2_in = info.get("pts2_in", None)
            inlier_matches = info.get("matches_inliers_all", None)

            if (pts1_in is not None) and (pts2_in is not None) and (pts2_in.shape[0] >= 8):
                # _triangulate_and_filter must return: (new_pts_world, good_mask)
                new_pts_world, good_mask = self._triangulate_and_filter(
                    prev.pose, frame.pose, pts1_in, pts2_in
                )

                if new_pts_world.shape[0] > 0 and inlier_matches is not None:
                    kept_matches = [m for m, keep in zip(inlier_matches, good_mask) if keep]

                    # Add points manually so we also save observations for Part 6
                    for Xw, m in zip(new_pts_world, kept_matches):
                        pt = Point3D(Xw, self.map._next_pid)
                        pt.observations.append((prev.id, m.queryIdx))
                        pt.observations.append((frame.id, m.trainIdx))
                        self.map.points.append(pt)
                        self.map._next_pid += 1

                    # ---------------- Part 5: update PnP DB correctly ----------------
                    if frame.descriptors is not None:
                        des_in = np.asarray([frame.descriptors[m.trainIdx] for m in kept_matches])

                        if des_in.shape[0] == new_pts_world.shape[0]:
                            self.map_db_xyz = np.vstack([
                                self.map_db_xyz,
                                new_pts_world.astype(np.float32)
                            ])

                            if self.map_db_des is None:
                                self.map_db_des = des_in.copy()
                            else:
                                self.map_db_des = np.vstack([self.map_db_des, des_in])

                    if i % 10 == 0:
                        total_pts = self.map.points_array().shape[0]
                        print(f"[Frame {i}] Triangulated={new_pts_world.shape[0]}  MapTotal={total_pts}")

            # ---------------- Part 5: PnP relocalization every N frames ----------------
            if (i % pnp_interval == 0) and (self.map_db_des is not None) and (self.map_db_xyz.shape[0] >= min_pnp_corr):
                if frame.descriptors is not None and frame.descriptors.shape[0] > 0:
                    knn = pnp_matcher.knnMatch(frame.descriptors, self.map_db_des, k=2)

                    good = []
                    for m_n in knn:
                        if len(m_n) < 2:
                            continue
                        m, n = m_n
                        if m.distance < ratio * n.distance:
                            good.append(m)

                    if len(good) >= min_pnp_corr:
                        img_pts = np.float32([frame.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 2)
                        obj_pts = np.float32([self.map_db_xyz[m.trainIdx] for m in good]).reshape(-1, 3)

                        ok, rvec, tvec, inl = cv2.solvePnPRansac(
                            objectPoints=obj_pts,
                            imagePoints=img_pts,
                            cameraMatrix=self.k,
                            distCoeffs=None,
                            reprojectionError=pnp_reproj_err,
                            confidence=0.999,
                            iterationsCount=200
                        )

                        if ok and inl is not None and len(inl) >= min_pnp_corr:
                            Rcw, _ = cv2.Rodrigues(rvec)   # world -> camera
                            Tcw = np.eye(4)
                            Tcw[:3, :3] = Rcw
                            Tcw[:3, 3] = tvec.reshape(3)

                            # convert to Twc (camera -> world)
                            frame.pose = np.linalg.inv(Tcw)
                            print(f"[Frame {i}] PnP relocalization OK: inliers={len(inl)}")

            
            # ---------------- Part 8: Loop Closure Detection ----------------
            if i % 60 == 0 and i > 100:  # Check every 60 frames, after frame 100
                is_loop, loop_frame_id = self.detect_loop_closure(i, min_frame_gap=100, min_matches=80)
                
                if is_loop:
                    print(f"[LOOP CLOSURE] Saving state before correction...")
                    
                    # Save trajectory BEFORE correction
                    traj_before = np.array([fr.pose[:3, 3].copy() for fr in self.frames])
                    
                    # Measure drift BEFORE
                    drift_before = np.linalg.norm(traj_before[-1] - traj_before[0])
                    
                    # Calculate loop closure error (distance between matched frames)
                    loop_error_before = np.linalg.norm(
                        self.frames[i].pose[:3, 3] - self.frames[loop_frame_id].pose[:3, 3]
                    )
                    
                    print(f"[LOOP CLOSURE] BEFORE optimization:")
                    print(f"  - Start-to-end drift: {drift_before:.2f} units")
                    print(f"  - Loop closure error (frame {i} ↔ {loop_frame_id}): {loop_error_before:.2f} units")
                    print(f"  - Total map points: {len(self.map.points)}")
                    
                    # Run optimization (limited window for speed)
                    print(f"[LOOP CLOSURE] Running loop-closure pose optimization on frames {loop_frame_id}→{i}...")
                    self.correct_loop_closure(i, loop_frame_id, window_size=150)                    
                    # Rebuild trajectory AFTER correction
                    self.trajectory = [fr.pose[:3, 3].copy() for fr in self.frames]
                    traj_after = np.array(self.trajectory)
                    did_optimize = True
                    
                    # Measure drift AFTER
                    drift_after = np.linalg.norm(traj_after[-1] - traj_after[0])
                    
                    # Calculate loop closure error AFTER
                    loop_error_after = np.linalg.norm(
                        self.frames[i].pose[:3, 3] - self.frames[loop_frame_id].pose[:3, 3]
                    )
                    
                    # Calculate improvements
                    drift_improvement = ((drift_before - drift_after) / drift_before * 100) if drift_before > 0 else 0
                    loop_improvement = ((loop_error_before - loop_error_after) / loop_error_before * 100) if loop_error_before > 0 else 0
                    
                    # Calculate trajectory change magnitude
                    traj_change = np.mean(np.linalg.norm(traj_after - traj_before, axis=1))
                    
                    print(f"\n[LOOP CLOSURE] AFTER optimization:")
                    print(f"  - Start-to-end drift: {drift_after:.2f} units (Δ {drift_improvement:+.1f}%)")
                    print(f"  - Loop closure error: {loop_error_after:.2f} units (Δ {loop_improvement:+.1f}%)")
                    print(f"  - Average trajectory shift: {traj_change:.2f} units")
                    print(f"  - Total map points: {len(self.map.points)}")
                    
                    print(f"\n[LOOP CLOSURE] EFFECT SUMMARY:")
                    print(f"  ✓ Drift reduction: {abs(drift_after - drift_before):.2f} units")
                    print(f"  ✓ Loop closure improvement: {abs(loop_error_after - loop_error_before):.2f} units")
                    print(f"  ✓ Trajectory corrected across {len(self.frames)} frames")
                    print(f"{'='*60}\n")
                    
            # ---------------- Part 6 + Part 7: reprojection error and optimization 
            
            if i % 50 == 0 and i > 0:  # Optimize every 50 frames (skip frame 0)
                rep_before = self.reprojection_error_stats()
                print(
                    f"[Frame {i}] Reprojection BEFORE opt (px): "
                    f"n={rep_before['n']} mean={rep_before['mean']:.2f} med={rep_before['median']:.2f} "
                    f"rmse={rep_before['rmse']:.2f} p95={rep_before['p95']:.2f}"
                )

                if rep_before["n"] > 0:
                    # Windowed optimization: only optimize last 20 frames
                    self.optimize_poses(window_size=20)
                    did_optimize = True
            
                    rep_after = self.reprojection_error_stats()
                    print(
                        f"[Frame {i}] Reprojection AFTER  opt (px): "
                        f"n={rep_after['n']} mean={rep_after['mean']:.2f} med={rep_after['median']:.2f} "
                        f"rmse={rep_after['rmse']:.2f} p95={rep_after['p95']:.2f}"
                    )

           # ---------------- trajectory / visualization ----------------
            # Only rebuild trajectory after optimization, otherwise just update current position
            if did_optimize:
                # Just optimized - rebuild entire trajectory from updated poses
                self.trajectory = [fr.pose[:3, 3].copy() for fr in self.frames]
                print(f"  [Traj] Rebuilt trajectory after optimization")
            else:
                # Normal frame - ALWAYS APPEND (never assign by index)
                pos = frame.pose[:3, 3]
                self.trajectory.append(pos.copy())

            traj = np.array(self.trajectory)
            if self.smooth:
                traj = self.smooth_traj(traj)

            pts = self.map.points_array()
            self.visualizer.update(traj, pts, i, total_frames)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                raise SystemExit

        traj = np.array(self.trajectory)
        if self.smooth:
            traj = self.smooth_traj(traj)

        frame_count = len(self.images)
        pts = self.map.points_array()
        self.visualizer.update(traj, pts, frame_count, total_frames)

        while True:
            pts = self.map.points_array()
            self.visualizer.update(traj, pts, frame_count, total_frames)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

        cv2.destroyAllWindows()