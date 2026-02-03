import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import time


class VOVisualizer:    
    def __init__(self, vo_system):
        self.vo = vo_system
        self.trajectory_points = []
        self.current_frame_idx = 0
        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)
        self.setup_3d_plot()
        
    def setup_3d_plot(self):
        self.ax_3d.set_xlabel('X (Left/Right)')
        self.ax_3d.set_ylabel('Y (Forward/Backward)')
        self.ax_3d.set_zlabel('Z (Up/Down)')
        self.ax_3d.set_title('3D Trajectory')
        self.ax_3d.grid(True)
        self.ax_3d.view_init(elev=20, azim=-45)
        
    def draw_camera(self, position, rotation, scale=0.5, color='red'):
        frustum = np.array([[0,0,0], [-0.5,-0.5,1], [0.5,-0.5,1], [0.5,0.5,1], [-0.5,0.5,1]]) * scale
        frustum_world = (rotation @ frustum.T).T + position
        
        for i in range(1, 5):
            self.ax_3d.plot3D([frustum_world[0,0], frustum_world[i,0]], 
                             [frustum_world[0,1], frustum_world[i,1]], 
                             [frustum_world[0,2], frustum_world[i,2]], color=color, linewidth=1)
        
        for i in range(1, 5):
            next_i = i + 1 if i < 4 else 1
            self.ax_3d.plot3D([frustum_world[i,0], frustum_world[next_i,0]], 
                             [frustum_world[i,1], frustum_world[next_i,1]], 
                             [frustum_world[i,2], frustum_world[next_i,2]], color=color, linewidth=1)
    
    def update_3d_plot(self):
        self.ax_3d.cla()
        self.setup_3d_plot()
        
        if len(self.trajectory_points) > 0:
            trajectory = np.array(self.trajectory_points)
            
            if len(self.trajectory_points) >= 11:
                smoothed = self.vo.smooth_trajectory(window_length=11, polyorder=2)
                self.ax_3d.plot3D(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], 
                                'b-', linewidth=2, label='Smoothed Trajectory')
                current_pos = smoothed[-1]
            else:
                self.ax_3d.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                                'b-', linewidth=2, label='Trajectory')
                current_pos = trajectory[-1]
            
            self.ax_3d.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                            c='green', s=100, marker='o', label='Start')
            
            if len(trajectory) > 1:
                self.ax_3d.scatter(current_pos[0], current_pos[1], current_pos[2], 
                                c='red', s=100, marker='o', label='Current')
                self.draw_camera(current_pos, self.vo.current_rotation, scale=0.3, color='red')
            
            max_range = np.array([trajectory[:, 0].max() - trajectory[:, 0].min(),
                                trajectory[:, 1].max() - trajectory[:, 1].min(),
                                trajectory[:, 2].max() - trajectory[:, 2].min()]).max() / 2.0
            
            mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) * 0.5
            mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) * 0.5
            mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) * 0.5
            
            self.ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
            self.ax_3d.legend()
    
    def show_keypoints(self, frame, keypoints):
        img_with_kp = cv2.drawKeypoints(frame.image, keypoints, None, color=(0, 255, 0),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_rgb = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)
        
        self.ax_2d.cla()
        self.ax_2d.imshow(img_rgb)
        self.ax_2d.set_title(f'Frame {frame.id} - {len(keypoints)} Keypoints')
        self.ax_2d.axis('off')
    
    def run_realtime(self, delay_ms=50):
        print("\n" + "="*60)
        print("Starting visual odometry")
        print("="*60 + "\n")
        
        self.vo.frames[0].extract_features(self.vo.detector, self.vo.feature_type)
        initial_position = self.vo.current_translation.flatten()
        self.trajectory_points.append(initial_position.copy())
        
        self.show_keypoints(self.vo.frames[0], self.vo.frames[0].keypoints)
        self.update_3d_plot()
        plt.pause(0.1)
        
        for i in range(len(self.vo.frames) - 1):
            if not plt.fignum_exists(self.fig.number):
                break
            
            frame1 = self.vo.frames[i]
            frame2 = self.vo.frames[i + 1]
            
            print(f"Processing frame pair {i+1}/{len(self.vo.frames)-1}...")
            
            self.vo.process_frame_pair(frame1, frame2)

            position = self.vo.current_translation.flatten()
            self.trajectory_points.append(position.copy())
            self.show_keypoints(frame2, frame2.keypoints)
            self.update_3d_plot()
            plt.pause(delay_ms / 1000.0)
        
        print("\nProcessing complete!")
        plt.ioff()
        plt.show()