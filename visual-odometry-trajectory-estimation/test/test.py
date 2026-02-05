import os
import argparse
from PIL import Image


def compress_inplace(folder, quality=70, max_width=None):
    for name in os.listdir(folder):
        if not name.lower().endswith((".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, name)

        img = Image.open(path).convert("RGB")

        # Optional resize (keep aspect ratio)
        if max_width is not None:
            w, h = img.size
            if w > max_width:
                scale = max_width / float(w)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Overwrite original file
        img.save(path, "JPEG", quality=quality, optimize=True)

        print(f"Replaced: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing JPG images")
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality (1–95)")
    parser.add_argument("--max_width", type=int, default=None, help="Resize if wider than this")

    args = parser.parse_args()

    compress_inplace(args.folder, args.quality, args.max_width)
