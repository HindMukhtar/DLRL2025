import os
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

def extract_number(filename):
    """Extract the number from filename for proper sorting"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def create_gif(image_folder, output_gif, duration=500, resize_to=None):
    images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
    if not images:
        print("No PNG images found in the specified folder.")
        return

    images.sort(key=extract_number)
    print(f"Found {len(images)} images, processing in order: {images[:5]}...")

    frames = []
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = Image.open(image_path)
        if resize_to:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)
        frames.append(img.convert('P', palette=Image.ADAPTIVE))  # Reduce color depth

    frames[0].save(
        output_gif,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,
        optimize=True  # Try to optimize GIF
    )
    print(f"GIF saved as {output_gif}")



def display_frames_as_animation(image_folder, interval=10):
    # Get all PNG files in the folder, sorted by number in filename
    images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
    if not images:
        print("No PNG images found in the specified folder.")
        return

    # Sort by the number in the filename
    images.sort(key=extract_number)

    # Load images as numpy arrays
    frames = [np.array(Image.open(os.path.join(image_folder, img))) for img in images]

    fig = plt.figure(figsize=(frames[0].shape[1]/80, frames[0].shape[0]/80), dpi=80)
    plt.axis('off')
    imgs = [[plt.imshow(f, animated=True)] for f in frames]
    ani = animation.ArtistAnimation(fig, imgs, interval=interval, blit=True)
    plt.close(fig)  # Avoid displaying the static plot
    display(HTML(ani.to_jshtml()))

if __name__ == "__main__":
    image_folder = "simulationImages/PPO"
    output_gif = "satellite_simulation_ppo.gif"
    # Example: resize to 640x480
    create_gif(image_folder, output_gif, duration=500, resize_to=(640, 480))
    display_frames_as_animation("simulationImages/PPO", interval=40)

