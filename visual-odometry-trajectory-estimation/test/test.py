# from PIL import Image
# import pillow_heif
# import glob
# import os

# # Register HEIF opener
# pillow_heif.register_heif_opener()

# input_folder = "./test_set"

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


import pangolin
import OpenGL.GL as gl
pangolin.CreateWindowAndBind ('Test Window', 640 , 480)
gl.glClearColor(0.2 , 0.2 , 0.2 , 1.0)
while not pangolin.ShouldQuit():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
pangolin.FinishFrame()
