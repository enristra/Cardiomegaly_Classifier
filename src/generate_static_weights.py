import numpy as np
import matplotlib.pyplot as plt
import os

def generate_static_weight(shape, radius, shift_x_percent=-0.2):
    """Genera una matrice 2D con pesi statici centrati in un cerchio"""
    height, width = shape
    weights = np.zeros((height, width), dtype=np.float32)
    shift_x=int(shift_x_percent * width)
    center_x=width // 2 + shift_x
    center_y = height // 2


    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    weights[dist_from_center <= radius] = 1.0
    return weights

def save_weights(weights, res, out_dir="/Users/Enrico/Desktop/projects/Cardiomegaly_Classifier/data/weights"):
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, f"static_weights_{res}x{res}.npy")
    png_path = os.path.join(out_dir, f"static_weights_{res}x{res}.png")

    np.save(npy_path, weights)

    plt.imshow(weights, cmap="hot")
    plt.title(f"Static weights {res}x{res}")
    plt.axis("off")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    print(f"[✔] Saved: {npy_path}, {png_path}")

if __name__ == "__main__":
    resolutions = {
        224: 40,
        128: 25,
        64: 12,
    }

    for res, radius in resolutions.items():
        weights = generate_static_weight((res, res), radius, shift_x_percent=-0.2)
        save_weights(weights, res)
