import numpy as np
import cv2
import pypangolin as pangolin
import OpenGL.GL as gl
from typing import List, Tuple
import time


class VOVisualizer:    
    def __init__(self, vo_system):
        self.vo = vo_system
        self.trajectory_points = []
        self.current_frame_idx = 0
        self.window_width = 1280
        self.window_height = 720
        
    def init_pangolin(self):
        pangolin.CreateWindowAndBind('Visual Odometry', self.window_width, self.window_height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.window_width, self.window_height, 420, 420,
                                     self.window_width // 2, self.window_height // 2, 0.1, 10000),
            pangolin.ModelViewLookAt(0, -50, -50, 0, 0, 0, pangolin.AxisY)
        )
        
        self.handler = pangolin.Handler3D(self.scam)
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(pangolin.Attach(0), pangolin.Attach(1), 
                           pangolin.Attach(0), pangolin.Attach(1), 
                           -self.window_width / self.window_height)
        self.dcam.SetHandler(self.handler)
        
    def draw_camera(self, pose, scale=1.0):
        w = scale * 0.5
        h = scale * 0.3
        z = scale * 0.6
        
        vertices = np.array([
            [0, 0, 0],
            [-w, -h, z],
            [w, -h, z],
            [w, h, z],
            [-w, h, z],
        ])
        
        R = pose[:3, :3]
        t = pose[:3, 3]
        vertices_world = (R @ vertices.T).T + t
        
        gl.glColor3f(0.0, 0.5, 1.0)
        gl.glLineWidth(2)
        
        gl.glBegin(gl.GL_LINES)
        for i in range(1, 5):
            gl.glVertex3f(*vertices_world[0])
            gl.glVertex3f(*vertices_world[i])
        
        for i in range(1, 5):
            gl.glVertex3f(*vertices_world[i])
            gl.glVertex3f(*vertices_world[(i % 4) + 1])
        gl.glEnd()
        
    def draw_trajectory(self, positions):
        if len(positions) < 2:
            return
        
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glLineWidth(3)
        
        gl.glBegin(gl.GL_LINE_STRIP)
        for pos in positions:
            gl.glVertex3f(pos[0], pos[1], pos[2])
        gl.glEnd()
        
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        for pos in positions:
            gl.glVertex3f(pos[0], pos[1], pos[2])
        gl.glEnd()
        
    def draw_grid(self, size=50, spacing=5.0):
        gl.glColor3f(0.3, 0.3, 0.3)
        gl.glLineWidth(1)
        gl.glBegin(gl.GL_LINES)
        for i in range(-size, size + 1):
            gl.glVertex3f(-size * spacing, 0, i * spacing)
            gl.glVertex3f(size * spacing, 0, i * spacing)
            gl.glVertex3f(i * spacing, 0, -size * spacing)
            gl.glVertex3f(i * spacing, 0, size * spacing)
        gl.glEnd()
    
    def run_realtime(self, delay_ms=50):
        print("\n" + "="*60)
        print("Starting visual odometry")
        print("="*60 + "\n")
        
        self.init_pangolin()
        
        self.vo.frames[0].extract_features(self.vo.detector, self.vo.feature_type)
        initial_position = self.vo.current_translation.flatten()
        self.trajectory_points.append(initial_position.copy())
        
        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.95, 0.95, 0.95, 1.0)
            
            self.dcam.Activate(self.scam)
            
            self.draw_grid(size=20, spacing=5.0)
            
            if self.current_frame_idx < len(self.vo.frames) - 1:
                frame1 = self.vo.frames[self.current_frame_idx]
                frame2 = self.vo.frames[self.current_frame_idx + 1]
                
                print(f"Processing frame pair {self.current_frame_idx+1}/{len(self.vo.frames)-1}...")
                
                self.vo.process_frame_pair(frame1, frame2)

                position = self.vo.current_translation.flatten()
                self.trajectory_points.append(position.copy())
                
                img_with_kp = cv2.drawKeypoints(frame2.image, frame2.keypoints, None,
                                               color=(0, 255, 0),
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                display_img = cv2.resize(img_with_kp, (640, 480))
                cv2.imshow('Current Frame - Keypoints', display_img)
                
                self.current_frame_idx += 1
                time.sleep(delay_ms / 1000.0)
            
            if len(self.trajectory_points) > 1:
                trajectory_array = np.array(self.trajectory_points)
                
                if len(self.trajectory_points) >= 11:
                    smoothed = self.vo.smooth_trajectory(window_length=11, polyorder=2)
                    self.draw_trajectory(smoothed)
                else:
                    self.draw_trajectory(trajectory_array)
                
                if self.current_frame_idx > 0 and self.current_frame_idx <= len(self.vo.frames):
                    current_frame = self.vo.frames[self.current_frame_idx - 1]
                    self.draw_camera(current_frame.pose, scale=2.0)
            
            pangolin.FinishFrame()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
        print("\nProcessing complete!")