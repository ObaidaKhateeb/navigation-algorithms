import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_ground_truth(path):
    positions = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            tx, ty, tz = map(float, parts[1:4])

            positions.append([tx, ty, tz])

    if len(positions) == 0:
        raise RuntimeError("No ground truth data found")

    return np.array(positions)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", help="Path to ground_truth.txt")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "b-", linewidth=2, label="Trajectory")

    ax.scatter(gt[0, 0], gt[0, 1], gt[0, 2],
               c="green", s=60, label="Start")

    ax.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2],
               c="red", s=60, label="End")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ground Truth Trajectory")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
