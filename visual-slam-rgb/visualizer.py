import numpy as np


class BaseVisualizer:
    def update(
        self,
        traj: np.ndarray,
        points=None,
        frame_num=0,
        total_frames=0
    ):
        pass

    def close(self):
        pass


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
            pangolin.ProjectionMatrix(
                1024, 768, 500, 500, 512, 389, 0.1, 1000
            ),
            pangolin.ModelViewLookAt(
                0, -9, -30, 0, 0, 0, 0, -1, 0
            )
        )
        self.handler = pangolin.Handler3D(self.s_cam)
        self.d_cam = pangolin.CreateDisplay()
        self.d_cam.SetBounds(
            pangolin.Attach(0.0), pangolin.Attach(1.0),
            pangolin.Attach(0.0), pangolin.Attach(1.0),
            -1024.0 / 768.0
        )
        self.d_cam.SetHandler(self.handler)

    def update(
        self, 
        traj: np.ndarray,
        points=None,
        frame_num=0,
        total_frames=0
    ):
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
            
        # Draw point cloud (Part 4)
        if points is not None:
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            if points.shape[0] > 0:
                gl.glPointSize(2)
                gl.glColor3f(0.0, 0.5, 1.0)
                gl.glBegin(gl.GL_POINTS)
                for X in points:
                    gl.glVertex3f(float(X[0]), float(X[1]), float(X[2]))
                gl.glEnd()

        # coloring the start and end points
        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(
            float(traj[0][0]), float(traj[0][1]),
            float(traj[0][2])
        )
        gl.glEnd()
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(
            float(traj[-1][0]), float(traj[-1][1]),
            float(traj[-1][2])
        )
        gl.glEnd()

        pangolin.FinishFrame()

    # Function responsible for making the camera follow
    # the trajectory (the last portion of it).
    # Handles the issue of the trajectory moving
    # outside the visible window.
    def update_cam_view(self, traj: np.ndarray):
        num_points = min(20, traj.shape[0])
        last_points = traj[-num_points:]
        center = np.mean(last_points, axis=0)
        self.s_cam.Follow(
            self.pangolin.OpenGlMatrix.Translate(
                center[0], center[1], center[2]
            )
        )

    def draw_grid_and_axes(self):
        gl = self.gl

        # grid draw
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

        # axes draw
        gl.glLineWidth(3)
        gl.glColor3f(1.0, 1.0, 1.0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(10.0, 0.0, 0.0)
        gl.glEnd()
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 10.0, 0.0)
        gl.glEnd()
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 10.0)
        gl.glEnd()
