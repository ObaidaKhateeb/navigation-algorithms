import os
import glob
import time
import argparse
import numpy as np
import cv2

import pypangolin as pangolin
from OpenGL.GL import *


# =========================
# Frame object
# =========================
class Frame:
    def __init__(self, idx, image):
        self.id = idx
        self.image = image
        self.kps = None
        self.des = None
        self.pose = np.eye(4, dtype=np.float64)


# =========================
# Utility functions
# =========================
def load_images(folder):
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, "*" + e)))
    return sorted(paths)


def build_K_from_image(img):
    h, w = img.shape[:2]
    f = max(h, w)
    return np.array([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)


# =========================
# Visual Odometry core
# =========================
class MonoVisualOdometry:
    def __init__(self, K, feature_type="ORB"):
        self.K = K
        self.T_wc = np.eye(4, dtype=np.float64)

        if feature_type == "ORB":
            self.detector = cv2.ORB_create(2000)
            self.norm = cv2.NORM_HAMMING
        else:
            self.detector = cv2.SIFT_create(2000)
            self.norm = cv2.NORM_L2

        self.matcher = cv2.BFMatcher(self.norm)

    def extract(self, frame: Frame):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        frame.kps, frame.des = self.detector.detectAndCompute(gray, None)

    def estimate_motion(self, f1: Frame, f2: Frame):
        if f1.des is None or f2.des is None:
            return None, None

        matches = self.matcher.knnMatch(f1.des, f2.des, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 8:
            return None, None

        pts1 = np.float32([f1.kps[m.queryIdx].pt for m in good])
        pts2 = np.float32([f2.kps[m.trainIdx].pt for m in good])

        E, _ = cv2.findEssentialMat(
            pts1, pts2, self.K,
            cv2.RANSAC, 0.999, 1.0
        )

        if E is None:
            return None, None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t

    def integrate(self, R, t, scale=1.0):
        t = t * scale

        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = (-R.T @ t).ravel()

        self.T_wc = self.T_wc @ T
        return self.T_wc


# =========================
# Pangolin Viewer
# =========================
class TrajectoryViewer:
    def __init__(self):
        self.poses = []

        pangolin.CreateWindowAndBind(
            "Visual Odometry Trajectory | X=Red Y=Green Z=Blue", 1024, 768
        )
        glEnable(GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin.ModelViewLookAt(0, -8, -8, 0, 0, 0, 0, 0, 1)
        )

        self.handler = pangolin.Handler3D(self.scam)
        self.dcam = pangolin.CreateDisplay()
        # self.dcam.SetBounds(0, 1, 0, 1, -1024 / 768)
        self.dcam.SetBounds(pangolin.Attach(0.0),pangolin.Attach(1.0),pangolin.Attach(0.0),pangolin.Attach(1.0),-1024.0 / 768.0)
        self.dcam.SetHandler(self.handler)

    def add_pose(self, T_wc):
        self.poses.append(T_wc.copy())

    def draw_axes(self, scale=5.0):
        glLineWidth(3)
        glBegin(GL_LINES)

        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(scale, 0, 0)

        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, scale, 0)

        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, scale)

        glEnd()

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.dcam.Activate(self.scam)

        self.draw_axes()

        if len(self.poses) > 1:
            glColor3f(1, 1, 0)
            glLineWidth(2)
            glBegin(GL_LINE_STRIP)
            for T in self.poses:
                glVertex3f(T[0, 3], T[1, 3], T[2, 3])
            glEnd()

        pangolin.FinishFrame()

    def should_close(self):
        return pangolin.ShouldQuit()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="ORB", choices=["ORB", "SIFT"])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--wait", type=int, default=1)
    args = parser.parse_args()

    img_paths = load_images("./test/right_test_simple2")
    if len(img_paths) < 2:
        raise RuntimeError("Dataset_VO must contain at least 2 images")

    first_img = cv2.imread(img_paths[0])
    K = build_K_from_image(first_img)

    vo = MonoVisualOdometry(K, args.feature)
    viewer = TrajectoryViewer()

    prev_frame = None
    cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)

    for i, path in enumerate(img_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        frame = Frame(i, img)
        vo.extract(frame)

        kp_vis = cv2.drawKeypoints(img, frame.kps, None)
        cv2.imshow("Keypoints", kp_vis)

        if prev_frame is not None:
            R, t = vo.estimate_motion(prev_frame, frame)
            if R is not None:
                pose = vo.integrate(R, t, args.scale)
                frame.pose = pose
            viewer.add_pose(vo.T_wc)
        else:
            viewer.add_pose(vo.T_wc)

        prev_frame = frame
        viewer.draw()

        if cv2.waitKey(args.wait) & 0xFF in [27, ord('q')]:
            break
        if viewer.should_close():
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()