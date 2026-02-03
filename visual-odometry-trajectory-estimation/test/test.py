# from PIL import Image
# import pillow_heif
# import glob
# import os

# # Register HEIF opener
# pillow_heif.register_heif_opener()

# input_folder = "./test/right_test"

# for img_path in glob.glob(os.path.join(input_folder, "*")):
#     # Skip if already proper JPG
#     if img_path.lower().endswith('.jpg') and 'HEIC' not in img_path.upper():
#         continue
    
#     try:
#         img = Image.open(img_path)
        
#         # Convert to RGB if needed
#         if img.mode in ("RGBA", "LA", "P"):
#             img = img.convert("RGB")
        
#         # Get base filename (remove .HEIC.jpg or just .HEIC)
#         base_name = os.path.basename(img_path)
#         base_name = base_name.replace('.HEIC.jpg', '').replace('.HEIC', '').replace('.heic', '')
        
#         output_path = os.path.join(input_folder, f"{base_name}.jpg")
        
#         img.save(output_path, "JPEG", quality=95)
#         print(f"✓ Converted: {base_name}.jpg")
        
#         # Remove original HEIC file
#         if img_path != output_path:
#             os.remove(img_path)
#             print(f"  Removed: {os.path.basename(img_path)}")
            
#     except Exception as e:
#         print(f"✗ Error converting {os.path.basename(img_path)}: {e}")

# print("\nConversion complete!")


# import pangolin
# import OpenGL.GL as gl
# pangolin.CreateWindowAndBind ('Test Window', 640 , 480)
# gl.glClearColor(0.2 , 0.2 , 0.2 , 1.0)
# while not pangolin.ShouldQuit():
#     gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
# pangolin.FinishFrame()


from PIL import Image
import os
import sys

# --- 1. Configuration ---
output_folder = "sliding_3_different_objects"
num_frames = 50
# Canvas size (Width, Height)
canvas_size = (700, 500)  
# Target size for all objects so they look uniform
object_target_size = (100, 100) 
bg_color = (255, 255, 255) # White background

# The list of required input files
required_files = ["source_ball1.png", "source_ball2.png", "source_ball3.png"]
# A list to store the processed images ready for pasting
loaded_objects = []


# --- 2. Setup and Loading Loop ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Checking and loading source images...")

# Loop through the 3 required filenames
for fname in required_files:
    # Check if file exists before trying to open
    if not os.path.exists(fname):
        print(f"\nERROR: Missing file! Could not find '{fname}'")
        print("Please ensure all 3 source_ball files are in the same folder.")
        sys.exit()

    try:
        # Load image and ensure RGBA for transparency
        raw_img = Image.open(fname).convert("RGBA")
        # Resize uniformly
        resized_img = raw_img.resize(object_target_size, Image.Resampling.LANCZOS)
        # Add the finished image to our list
        loaded_objects.append(resized_img)
        print(f" - Successfully loaded and resized: {fname}")
    except Exception as e:
        print(f"Error processing image {fname}: {e}")
        sys.exit()


# --- 3. Position Calculations ---
# (This logic is identical to the previous script to ensure even spacing)

# A. Horizontal Movement (X-axis)
start_x = -object_target_size[0]
end_x = canvas_size[0]
total_distance = end_x - start_x
pixels_per_frame = total_distance / num_frames

# B. Vertical Positions (Y-axis)
canvas_h = canvas_size[1]
obj_h = object_target_size[1]

if canvas_h < (3 * obj_h):
     print("Error: Canvas height is too small for 3 stacked objects.")
     sys.exit()

total_used_height = 3 * obj_h
remaining_space = canvas_h - total_used_height
padding = remaining_space // 4

# Calculate the fixed Y positions for the three slots
top_y = padding
middle_y = padding + obj_h + padding
bottom_y = middle_y + obj_h + padding


# --- 4. Generate Frames ---
print(f"\nGenerating {num_frames} frames...")

for i in range(num_frames):
    bg_img = Image.new('RGB', canvas_size, bg_color)
    current_x = int(start_x + (i * pixels_per_frame))
    
    # --- PASTE THE 3 DIFFERENT OBJECTS ---
    # We access the images from the loaded_objects list by index [0], [1], [2]
    
    # Paste Object 1 (Top)
    # Remember the 3rd argument is the mask for transparency
    bg_img.paste(loaded_objects[0], (current_x, top_y), loaded_objects[0])
    
    # Paste Object 2 (Middle)
    bg_img.paste(loaded_objects[1], (current_x, middle_y), loaded_objects[1])
    
    # Paste Object 3 (Bottom)
    bg_img.paste(loaded_objects[2], (current_x, bottom_y), loaded_objects[2])
    
    # Save frame
    filename = f"{output_folder}/frame_{i+1:02d}.jpg"
    bg_img.save(filename, quality=95)
    
    if (i+1) % 10 == 0:
        print(f"Saved frame {i+1}/{num_frames}...")

print(f"Done! Check the '{output_folder}' folder.")