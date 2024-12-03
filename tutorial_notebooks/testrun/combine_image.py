from PIL import Image

def combine_images(images, output_path):
    # Ensure there are exactly 9 images
    if len(images) != 9:
        raise ValueError("You must provide exactly 9 images.")
    
    # Open all images and ensure they have the same size
    opened_images = [Image.open(img) for img in images]
    widths, heights = zip(*(img.size for img in opened_images))
    
    # Use the smallest width and height to resize all images
    min_width, min_height = min(widths), min(heights)
    resized_images = [img.resize((min_width, min_height)) for img in opened_images]
    
    # Create a new blank image with the size for the 3x3 grid
    grid_width = 3 * min_width
    grid_height = 3 * min_height
    grid_image = Image.new("RGB", (grid_width, grid_height))
    
    # Paste images into the grid
    for idx, img in enumerate(resized_images):
        row, col = divmod(idx, 3)
        grid_image.paste(img, (col * min_width, row * min_height))
    
    # Save the resulting image
    grid_image.save(output_path)
    print(f"Combined image saved to {output_path}")

# Example usage
image_files = [
    "XXXXX.png", "XYXYX.png", "XZXZX.png",
    "YXYXY.png", "YYYYY.png", "YZYZY.png",
    "ZXZXZ.png", "ZYZYZ.png", "ZZZZZ.png",
]

output_file = "combined_image.jpg"
combine_images(image_files, output_file)
