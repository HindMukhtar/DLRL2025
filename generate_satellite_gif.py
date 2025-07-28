import os
from PIL import Image

def create_gif(image_folder, output_gif, duration=500):
    # Get all PNG files in the folder, sorted by name
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith('.png')]
    if not images:
        print("No PNG images found in the specified folder.")
        return

    frames = []
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frames.append(Image.open(image_path))

    # Save as GIF
    frames[0].save(
        output_gif,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )
    print(f"GIF saved as {output_gif}")

if __name__ == "__main__":
    image_folder = "simulationImages"
    output_gif = "satellite_simulation.gif"
    create_gif(image_folder, output_gif, duration=500)