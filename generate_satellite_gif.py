import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

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


def display_frames_as_animation(image_folder, interval=40):
    # Get all PNG files in the folder, sorted by name
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith('.png')]
    if not images:
        print("No PNG images found in the specified folder.")
        return

    # Load images as numpy arrays
    frames = [np.array(Image.open(os.path.join(image_folder, img))) for img in images]

    fig = plt.figure(figsize=(frames[0].shape[1]/80, frames[0].shape[0]/80), dpi=80)
    plt.axis('off')
    imgs = [[plt.imshow(f, animated=True)] for f in frames]
    ani = animation.ArtistAnimation(fig, imgs, interval=interval, blit=True)
    plt.close(fig)  # Avoid displaying the static plot
    display(HTML(ani.to_jshtml()))

if __name__ == "__main__":
    image_folder = "simulationImages"
    output_gif = "satellite_simulation.gif"
    create_gif(image_folder, output_gif, duration=500)
    display_frames_as_animation("simulationImages", interval=40)