# Visual Odometry - Drone Trajectory Estimation

## Team Members
- Obaida Khateeb, 201278066
- Falah Abu Raya, 212530034
- Lama Hammoud, 324839984

## Usage
```
python main.py dataset_path [-h] [--matplotlib] [--sift]
```

**Default behavior**: ORB, Pangolin, and 50ms delay between frames.

### Positional Arguments:
- `dataset_path` - Path to the dataset folder containing images

### Optional Arguments:
- `-h, --help` - Show help message
- `--sift` - Use SIFT instead of ORB
- `--matplotlib` - Use matplotlib instead of Pangolin for visualization

## Controls
- **Mouse**: Rotate and zoom the 3D view
- **ESC** or **'q'**: Quit