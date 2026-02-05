import cv2
import numpy as np
import glob
import argparse
import os

class Frame:
    def __init__(self, img, idx):
        self.id = idx
        self.image = img
        self.keypoints = None
        self.descriptors = None
        self.pose = np.eye(4)
        self.processed = False

class VisualOdometry:
    def __init__(self, image_paths, visualizer, use_sift=False):
        self.images = image_paths
        self.use_sift = use_sift
        self.orb = cv2.ORB_create(200)
        self.sift = cv2.SIFT_create()
        self.visualizer = visualizer
        if self.use_sift:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.frames = []
        self.trajectory = []
        self.K = None

    def extract_features(self, frame):
        detector = self.sift if self.use_sift else self.orb
        kp, des = detector.detectAndCompute(frame.image, None)
        frame.keypoints = kp
        frame.descriptors = des

    def estimate_motion(self, f1, f2):
        # 1) Must have descriptors
        if f1.descriptors is None or f2.descriptors is None:
            return None, None, []

        # 2) Match
        matches = self.matcher.match(f1.descriptors, f2.descriptors)

        # 3) Need enough matches
        if matches is None or len(matches) < 8:
            return None, None, []

        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:2000]

        # 4) Build point arrays
        pts1 = np.float32([f1.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([f2.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # 5) Need enough points for Essential matrix
        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            return None, None, matches

        # 6) Compute Essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None or mask is None:
            return None, None, matches

        # 7) Recover pose using inliers
        inliers = mask.ravel().astype(bool)
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        if pts1_in.shape[0] < 8:
            return None, None, matches

        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.K)

        return R, t, matches

    def run(self):
        for i, path in enumerate(self.images):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            frame = Frame(img, i)

            if self.K is None:
                h, w = img.shape
                fx = fy = 0.8 * w
                cx = w / 2
                cy = h / 2
                self.K = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0,  0,  1]])

            self.extract_features(frame)
            self.frames.append(frame)
            
            kp_img = cv2.drawKeypoints(frame.image, frame.keypoints, None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Features (Current Frame)", kp_img)

            if i == 0:
                self.trajectory.append(np.array([0.0, 0.0, 0.0]))
                self.visualizer.update(np.array(self.trajectory))
                continue

            prev = self.frames[i - 1]
            R, t, matches = self.estimate_motion(prev, frame)

            # if pose estimation failed, skip this frame
            if R is None or t is None:
                continue

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            frame.pose = prev.pose @ np.linalg.inv(T)

            pos = frame.pose[:3, 3]
            self.trajectory.append(pos)

            # --- window 1: keypoints/matches ---
            vis = cv2.drawMatches(prev.image, prev.keypoints,
                                  frame.image, frame.keypoints,
                                  matches[:50], None)
            cv2.imshow("Keypoints / Matches", vis)

            # --- window 2: trajectory realtime ---
            self.visualizer.update(np.array(self.trajectory))

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                raise SystemExit

        traj = np.array(self.trajectory)
        while True:
            self.visualizer.update(traj)

            if hasattr(self.visualizer, "closed") and self.visualizer.closed:
                break

            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                break
        
        cv2.destroyAllWindows()


class BaseVisualizer:
    def update(self, traj: np.ndarray):
        pass
    def close(self):
        pass

class MatplotlibVisualizer(BaseVisualizer):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.line, = self.ax.plot([], [], [], 'b-')
        self.start = None
        self.end = None

        self.ax.set_xlabel("X (right)")
        self.ax.set_ylabel("Y (up)")
        self.ax.set_zlabel("Z (forward)")
        self.ax.set_title("Estimated Trajectory (Real-Time)")

    def update(self, traj: np.ndarray):
        if traj.shape[0] < 2:
            return
        self.line.set_data(traj[:, 0], traj[:, 1])
        self.line.set_3d_properties(traj[:, 2])

        if self.start is None:
            self.start = self.ax.scatter(traj[0,0], traj[0,1], traj[0,2], c='g', s=60)

        if self.end is not None:
            self.end.remove()
        self.end = self.ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='r', s=60)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self):
        self.plt.close('all')


class PangolinVisualizer(BaseVisualizer):
    def __init__(self):
        import pypangolin as pangolin
        import OpenGL.GL as gl
        self.pangolin = pangolin
        self.gl = gl
        self.closed = False

        pangolin.CreateWindowAndBind("Trajectory (Pangolin)", 1024, 768)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.s_cam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin.ModelViewLookAt(0, -9, -30, 0, 0, 0, 0, -1, 0)
        )

        self.handler = pangolin.Handler3D(self.s_cam)
        self.d_cam = pangolin.CreateDisplay()
        self.d_cam.SetBounds(pangolin.Attach(0.0),pangolin.Attach(1.0),pangolin.Attach(0.0),
            pangolin.Attach(1.0),-1024.0 / 768.0)
        self.d_cam.SetHandler(self.handler)

    def update(self, traj: np.ndarray):
        pangolin = self.pangolin
        gl = self.gl

        if pangolin.ShouldQuit():
            self.closed = True
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.d_cam.Activate(self.s_cam)

        if traj.shape[0] >= 2:
            self.update_cam_view(traj)
            gl.glColor3f(1.0, 1.0, 0.0)
            gl.glLineWidth(3)
            gl.glBegin(gl.GL_LINE_STRIP)
            for p in traj:
                gl.glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            gl.glEnd()

        #coloring the start and end points 
        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(float(traj[0][0]), float(traj[0][1]), float(traj[0][2]))
        gl.glEnd()
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(float(traj[-1][0]), float(traj[-1][1]), float(traj[-1][2]))
        gl.glEnd()

        pangolin.FinishFrame()

    #function responsibe for making the camera follow the trajectory (the last portion of it)
    #the function created to handle the issue of the trajectory being moved outside the visible window
    def update_cam_view(self, traj: np.ndarray):
            num_points = min(20, traj.shape[0])
            last_points = traj[-num_points:]
            center = np.mean(last_points, axis=0) #center of the recently added points
            self.s_cam.Follow(self.pangolin.OpenGlMatrix.Translate(center[0], center[1], center[2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Path to directory containing images")
    parser.add_argument("--sift", action="store_true", 
            help="Use SIFT instead of ORB (default: ORB)")
    parser.add_argument("--matplotlib", action="store_true",
                        help="Use matplotlib visualizer (default: Pangolin)")
    args = parser.parse_args()

    image_dir = args.image_dir

    images = sorted(
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.jpeg"))
    )

    if len(images) == 0:
        raise RuntimeError("No images found in directory")
    visualizer = MatplotlibVisualizer() if args.matplotlib else PangolinVisualizer()
    vo = VisualOdometry(images, visualizer, use_sift=args.sift)
    vo.run()