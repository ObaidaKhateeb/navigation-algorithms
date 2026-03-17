import numpy as np
from visualizer import BaseVisualizer


class MatplotlibVisualizer(BaseVisualizer):
    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig = plt.figure(facecolor="black")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.patch.set_facecolor("black")
        self.line, = self.ax.plot(
            [],
            [],
            [],
            "y-",
            linewidth=2,
        )
        self.start = None
        self.end = None

        self.ax.set_facecolor("black")
        self.ax.set_xlabel("X (right)", color="white")
        self.ax.set_ylabel("Y (up)", color="white")
        self.ax.set_zlabel("Z (forward)", color="white")
        self.ax.set_title("Estimated Trajectory (Real-Time)", color="white")
        self.ax.tick_params(colors="white")
        self.ax.grid(
            True, color="white", linestyle="-",
            linewidth=0.5, alpha=0.3
        )
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        self.point_cloud = None

        self.idx = None
        self.rng = np.random.default_rng(42)

    def update(
        self,
        traj: np.ndarray,
        points=None,
        frame_num=0,
        total_frames=0,
    ):
        if traj.shape[0] < 2:
            return
        self.line.set_data(traj[:, 0], traj[:, 1])
        self.line.set_3d_properties(traj[:, 2])

        # start and end points coloring
        if self.start is None:
            self.start = self.ax.scatter(
                traj[0, 0], traj[0, 1], traj[0, 2],
                c="g", s=60
            )
        if self.end is not None:
            self.end.remove()
        self.end = self.ax.scatter(
            traj[-1, 0], traj[-1, 1], traj[-1, 2],
            c="r", s=60
        )

        if points is not None:
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            if points.shape[0] > 0:
                if self.point_cloud is not None:
                    self.point_cloud.remove()
                # We take 500 random points and update them every few
                # frames because matplotlib gets slow with many points
                if self.idx is None or frame_num % 10 == 0:
                    self.idx = self.rng.choice(
                        points.shape[0],
                        min(500, points.shape[0]),
                        replace=False
                    )

                self.point_cloud = self.ax.scatter(
                    points[self.idx, 0],
                    points[self.idx, 1],
                    points[self.idx, 2],
                    c="cyan", s=1, alpha=0.4
                )

        self.ax.set_title(
            f"Estimated Trajectory "
            f"(Frame: {frame_num}/{total_frames})",
            color="white"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self):
        self.plt.close("all")
