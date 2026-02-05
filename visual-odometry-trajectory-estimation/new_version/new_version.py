import cv2
import numpy as np
import glob
import argparse
import os
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
        self.k = None

    def extract_features(self, frame):
        detector = self.sift if self.use_sift else self.orb
        kp, des = detector.detectAndCompute(frame.image, None)
        frame.keypoints = kp
        frame.descriptors = des

    def estimate_motion(self, f1, f2):
        #ensure descriptors exists
        if f1.descriptors is None or f2.descriptors is None:
            return None, None, []

        matches = self.matcher.match(f1.descriptors, f2.descriptors) #matching

        #check if there enough matches
        if matches is None or len(matches) < 8:
            return None, None, []

        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:2000]

        #point arrays building
        pts1 = np.float32([f1.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([f2.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        #Essential matrix computation 
        if pts1.shape[0] < 8 or pts2.shape[0] < 8:
            return None, None, matches
        essential_matrix, mask = cv2.findEssentialMat(pts1, pts2, self.k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if essential_matrix is None or mask is None:
            return None, None, matches

        #Pose recovering
        inliers = mask.ravel().astype(bool)
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        if pts1_in.shape[0] < 8:
            return None, None, matches

        _, rotation, translation, _ = cv2.recoverPose(essential_matrix, pts1_in, pts2_in, self.k)

        return rotation, translation, matches

    def smooth_traj(self, traj):
        try:
            if traj.shape[0] < 12: #smaller than the window size 
                return traj

            sm = traj.copy()
            for d in range(3):
                sm[1:, d] = savgol_filter(traj[1:, d], window_length=11, polyorder=3, mode="interp")
                
            return sm
        except Exception:
            return traj

    def run(self):
        for i, path in enumerate(self.images):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            frame = Frame(img, i)

            if self.k is None:
                h, w = img.shape
                fx = fy = 0.8 * w
                cx = w / 2
                cy = h / 2
                self.k = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0,  0,  1]])

            self.extract_features(frame)
            self.frames.append(frame)
            
            kp_img = cv2.drawKeypoints(frame.image, frame.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Features (Current Frame)", kp_img)

            if i == 0:
                self.trajectory.append(np.array([0.0, 0.0, 0.0]))
                traj = self.smooth_traj(np.array(self.trajectory))
                self.visualizer.update(traj)
                continue

            prev = self.frames[i - 1]
            rotation, translation, matches = self.estimate_motion(prev, frame)

            if rotation is None or translation is None:
                continue #if pose estimation failed, the frame is skipped

            frame.rotation_matrix = rotation
            frame.translation_vector = translation

            t_matrix = np.eye(4)
            t_matrix[:3, :3] = frame.rotation_matrix
            t_matrix[:3, 3] =  frame.translation_vector.flatten()

            frame.pose = prev.pose @ np.linalg.inv(t_matrix)

            pos = frame.pose[:3, 3]
            self.trajectory.append(pos)

            #keypoints/matches window
            vis = cv2.drawMatches(prev.image, prev.keypoints, frame.image, frame.keypoints, matches[:50], None)
            cv2.imshow("Keypoints / Matches", vis)

            #trajectory window
            traj = self.smooth_traj(np.array(self.trajectory))
            self.visualizer.update(traj)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                raise SystemExit

        traj = self.smooth_traj(np.array(self.trajectory))
        self.visualizer.update(traj)
        while True:
            self.visualizer.update(traj) #this intends to keep the trajectory window responsive at the end

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
        self.fig = plt.figure(facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.patch.set_facecolor('black')
        self.line, = self.ax.plot([], [], [], 'y-', linewidth = 2)
        self.start = None
        self.end = None

        self.ax.set_facecolor('black')
        self.ax.set_xlabel("X (right)", color='white')
        self.ax.set_ylabel("Y (up)", color='white')
        self.ax.set_zlabel("Z (forward)", color='white')
        self.ax.set_title("Estimated Trajectory (Real-Time)", color='white')
        self.ax.tick_params(colors='white')
        self.ax.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

    def update(self, traj: np.ndarray):
        if traj.shape[0] < 2:
            return
        self.line.set_data(traj[:, 0], traj[:, 1])
        self.line.set_3d_properties(traj[:, 2])

        #start and end points coloring 
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
        self.s_cam = pangolin.OpenGlRenderState(pangolin.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin.ModelViewLookAt(0, -9, -30, 0, 0, 0, 0, -1, 0))
        self.handler = pangolin.Handler3D(self.s_cam)
        self.d_cam = pangolin.CreateDisplay()
        self.d_cam.SetBounds(pangolin.Attach(0.0),pangolin.Attach(1.0),pangolin.Attach(0.0), pangolin.Attach(1.0),-1024.0 / 768.0)
        self.d_cam.SetHandler(self.handler)

    def update(self, traj: np.ndarray):
        pangolin = self.pangolin
        gl = self.gl

        if pangolin.ShouldQuit():
            self.closed = True
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.d_cam.Activate(self.s_cam)
        self.draw_grid_and_axes()

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

    def draw_grid_and_axes(self):
            gl = self.gl
            
            #grid draw
            gl.glColor3f(0.3, 0.3, 0.3)
            gl.glLineWidth(1)
            grid_size = 50
            grid_step = 5
            for i in range(-grid_size, grid_size + 1, grid_step):
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(float(i), 0.0, float(-grid_size))
                gl.glVertex3f(float(i), 0.0, float(grid_size))
                gl.glEnd()
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(float(-grid_size), 0.0, float(i))
                gl.glVertex3f(float(grid_size), 0.0, float(i))
                gl.glEnd()
            
            #axes draw 
            gl.glLineWidth(3)
            gl.glColor3f(1.0, 1.0, 1.0)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(0.0, 0.0, 0.0)
            gl.glVertex3f(10.0, 0.0, 0.0)
            gl.glEnd()
            gl.glColor3f(1.0, 1.0, 1.0)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(0.0, 0.0, 0.0)
            gl.glVertex3f(0.0, 10.0, 0.0)
            gl.glEnd()
            gl.glColor3f(1.0, 1.0, 1.0)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(0.0, 0.0, 0.0)
            gl.glVertex3f(0.0, 0.0, 10.0)
            gl.glEnd()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Path to directory containing images")
    parser.add_argument("--sift", action="store_true", help="Use SIFT instead of ORB (default: ORB)")
    parser.add_argument("--matplotlib", action="store_true", help="Use matplotlib visualizer (default: Pangolin)")
    args = parser.parse_args()

    image_dir = args.image_dir
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.jpeg")))

    if len(images) == 0:
        raise RuntimeError("No images found in directory")
    
    if args.matplotlib:
        visualizer = MatplotlibVisualizer()
    else:
        try:
            visualizer = PangolinVisualizer()
        except Exception as e:
            print("WARNING: pypangolin not found or the used one not compatible with the installed, using Matplotlib as visualizer instead.")
            visualizer = MatplotlibVisualizer()
    
    vo = VisualOdometry(images, visualizer, use_sift=args.sift)
    vo.run()