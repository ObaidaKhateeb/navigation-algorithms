#!/usr/bin/env python3
"""
Main script for Visual Odometry Drone Trajectory Estimation
Assignment 2 - Navigation, Mapping and Localization Algorithms
"""

import sys
import os
import argparse
from visual_odometry import VisualOdometry
from visualizer import VOVisualizer


def main():
    """Main entry point for the Visual Odometry system."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Visual Odometry for Drone Trajectory Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in batch mode (process all frames then visualize)
  python main.py /path/to/VO_Dataset
  
  # Run in real-time mode (process and visualize frame by frame)
  python main.py /path/to/VO_Dataset --realtime
  
  # Use SIFT instead of ORB features
  python main.py /path/to/VO_Dataset --sift
  
  # Adjust delay between frames in real-time mode
  python main.py /path/to/VO_Dataset --realtime --delay 200
        """
    )
    
    parser.add_argument('dataset_path', type=str,
                       help='Path to the VO_Dataset folder containing images')
    parser.add_argument('--realtime', action='store_true',
                       help='Run in real-time mode (process frames one by one)')
    parser.add_argument('--sift', action='store_true',
                       help='Use SIFT features instead of ORB')
    parser.add_argument('--delay', type=int, default=100,
                       help='Delay between frames in milliseconds (real-time mode only)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization (only run VO processing)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)
    
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Dataset path is not a directory: {args.dataset_path}")
        sys.exit(1)
    
    # Print header
    print("=" * 70)
    print("Visual Odometry - Drone Trajectory Estimation")
    print("Assignment 2 - Epipolar Visual Odometry")
    print("=" * 70)
    print(f"\nDataset: {args.dataset_path}")
    print(f"Feature detector: {'SIFT' if args.sift else 'ORB'}")
    print(f"Mode: {'Real-time' if args.realtime else 'Batch'}")
    if args.realtime:
        print(f"Frame delay: {args.delay}ms")
    print()
    
    try:
        # Initialize Visual Odometry system
        print("Initializing Visual Odometry system...")
        vo = VisualOdometry(args.dataset_path, use_sift=args.sift)
        
        # Load images
        vo.load_images()
        
        if args.no_viz:
            # Run without visualization
            print("\nRunning Visual Odometry (no visualization)...")
            vo.run()
            
            # Print results
            trajectory = vo.get_trajectory()
            print(f"\nTrajectory computed successfully!")
            print(f"Total frames: {len(vo.frames)}")
            print(f"Trajectory points: {len(trajectory)}")
            print(f"Start position: {trajectory[0]}")
            print(f"End position: {trajectory[-1]}")
            
            # Calculate trajectory statistics
            import numpy as np
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            total_distance = np.sum(distances)
            print(f"Total distance traveled: {total_distance:.2f} units")
            
        else:
            # Run with visualization
            visualizer = VOVisualizer(vo)
            
            if args.realtime:
                print("\n" + "=" * 70)
                print("REAL-TIME MODE")
                print("=" * 70)
                print("The system will process frames one by one and display:")
                print("  - Window 1: Current frame with detected keypoints")
                print("  - Window 2: 3D trajectory visualization (Pangolin)")
                print("\nControls:")
                print("  - Mouse: Rotate and zoom the 3D view")
                print("  - ESC or 'q': Quit")
                print("=" * 70)
                visualizer.run_realtime(delay_ms=args.delay)
            else:
                print("\n" + "=" * 70)
                print("BATCH MODE")
                print("=" * 70)
                print("The system will first process all frames, then display:")
                print("  - Window 1: Frames with keypoints (cycling)")
                print("  - Window 2: Complete 3D trajectory (Pangolin)")
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