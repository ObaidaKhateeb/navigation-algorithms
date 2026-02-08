"""
Visual Odometry Assignment

Students:
- Obaida Khateeb - 201278066
- Falah Abu Raya - 212530034
- Lama Hammoud - 324839984
"""

import glob
import argparse
import os
from visual_odometry import VisualOdometry
from visualizer_matplotlib import MatplotlibVisualizer
from visualizer import PangolinVisualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_dir", nargs="?", default=None,
        help="Path to directory containing images (default: Dataset_VO)"
    )
    parser.add_argument(
        "--sift", action="store_true",
        help="Use SIFT instead of ORB (default: ORB)"
    )
    parser.add_argument(
        "--matplotlib", action="store_true",
        help="Use matplotlib visualizer (default: Pangolin)"
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help=(
            "Smooth the displayed trajectory (visualization only). "
            "Kept optional to avoid hiding estimation and optimization effects."
        ),
    )
    
    args = parser.parse_args()

    if args.image_dir is None:
        default_dir = "Dataset_VO"
        if os.path.exists(default_dir) and os.path.isdir(default_dir):
            image_dir = default_dir
            print(f"No directory given, using default: {default_dir}")
        else:
            raise RuntimeError(
                f"No directory specified and default "
                f"directory '{default_dir}' not found"
            )
    else:
        image_dir = args.image_dir

    images = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )

    if not images:
        raise RuntimeError("No images found in directory")

    if args.matplotlib:
        visualizer = MatplotlibVisualizer()
    else:
        try:
            visualizer = PangolinVisualizer()
        except Exception:
            print(
                "WARNING: pypangolin not found or the used one "
                "not compatible with the installed, using "
                "Matplotlib as visualizer instead."
            )
            visualizer = MatplotlibVisualizer()

    vo = VisualOdometry(
        images,
        visualizer,
        use_sift=args.sift,
        smooth=args.smooth
    )
    vo.run()
