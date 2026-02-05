import glob
import argparse
import os
from visual_odometry import VisualOdometry
from visualizer_matplotlib import MatplotlibVisualizer
from visualizer import PangolinVisualizer

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