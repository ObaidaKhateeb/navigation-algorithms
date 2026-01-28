#!/usr/bin/env python3
import sys
import os
import argparse
from visual_odometry import VisualOdometry


def main():    
    parser = argparse.ArgumentParser(
        description='Visual Odometry for Drone Trajectory Estimation'
    )
    
    parser.add_argument('dataset_path', type=str,
                       help='Path to the VO_Dataset folder containing images')
    parser.add_argument('--realtime', action='store_true',
                       help='Run in real-time mode (process frames one by one)')
    parser.add_argument('--sift', action='store_true',
                       help='Use SIFT features instead of ORB')
    parser.add_argument('--delay', type=int, default=100,
                       help='Delay between frames in milliseconds (real-time mode only)')
    parser.add_argument('--matplotlib', action='store_true',
                       help='Use matplotlib instead of Pangolin for visualization')
    
    args = parser.parse_args()
    
    #dataset path validation 
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)
    
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Dataset path is not a directory: {args.dataset_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Visual Odometry - Drone Trajectory Estimation")
    print("=" * 70)
    print(f"\nDataset: {args.dataset_path}")
    print(f"Feature detector: {'SIFT' if args.sift else 'ORB'}")
    print(f"Mode: {'Real-time' if args.realtime else 'Batch'}")
    print(f"Visualizer: {'matplotlib' if args.matplotlib else 'Pangolin'}")
    if args.realtime:
        print(f"Frame delay: {args.delay}ms")
    print()
    
    try:
        print("Initializing Visual Odometry system...")
        vo = VisualOdometry(args.dataset_path, use_sift=args.sift)
        
        vo.load_images() #image loading

        #import appropriate visualizer
        if args.matplotlib:
            from visualizer_matplotlib import VOVisualizer
        else:
            from visualizer import VOVisualizer

        visualizer = VOVisualizer(vo) #initiallizing visualizer
        
        if args.realtime:
            print("\n" + "=" * 70)
            print("=" * 70)
            print("The system will process frames one by one and display:")
            print("\nControls:")
            print("  - Mouse: Rotate and zoom the 3D view")
            print("  - ESC or 'q': Quit")
            print("=" * 70)
            visualizer.run_realtime(delay_ms=args.delay)
        else:
            print("\n" + "=" * 70)
            print("=" * 70)
            print("The system will first process all frames, then display:")
            print("\nControls:")
            print("  - Mouse: Rotate and zoom the 3D view")
            print("  - ESC or 'q': Quit")
            print("=" * 70)
            visualizer.run_batch()
        
        print("\n" + "=" * 70)
        print("Visual Odometry completed successfully!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()