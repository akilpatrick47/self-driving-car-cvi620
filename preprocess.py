import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = 'data'          # folder that contains driving_log.csv and IMG/
IMG_HEIGHT = 66            # Nvidia model input height
IMG_WIDTH = 200            # Nvidia model input width
NUM_BINS = 25              # bins used to balance the steering histogram
MAX_SAMPLES_PER_BIN = 200  # max samples kept per bin (balance threshold)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir: str = DATA_DIR):
    """
    Load driving_log.csv and return image paths + steering angles.
    Only uses the center camera.

    Returns
    -------
    image_paths : list[str]
    steering    : np.ndarray  (float32)
    """
    csv_path = os.path.join(data_dir, 'driving_log.csv')
    df = pd.read_csv(csv_path, header=None,
                     names=['center', 'left', 'right',
                            'steering', 'throttle', 'brake', 'speed'])

    # Normalize paths so they work on any OS
    image_paths = []
    for path in df['center']:
        filename = os.path.basename(path.strip())
        image_paths.append(os.path.join(data_dir, 'IMG', filename))

    steering = df['steering'].values.astype(np.float32)

    print(f'[INFO] Loaded {len(image_paths)} samples from {csv_path}')
    return image_paths, steering


# ─────────────────────────────────────────────────────────────────────────────
# 2. Balance the dataset
# ─────────────────────────────────────────────────────────────────────────────
def balance_dataset(image_paths: list, steering: np.ndarray,
                    num_bins: int = NUM_BINS,
                    max_samples: int = MAX_SAMPLES_PER_BIN,
                    visualize: bool = True):
    """
    Undersample over-represented steering bins so the distribution is flat.
    Optionally plots a before/after histogram.

    Returns
    -------
    balanced_paths    : list[str]
    balanced_steering : np.ndarray
    """
    hist, bin_edges = np.histogram(steering, bins=num_bins)

    if visualize:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]),
                align='edge', color='steelblue', edgecolor='black')
        plt.axhline(max_samples, color='red', linestyle='--',
                    label=f'Threshold = {max_samples}')
        plt.title('Steering Angle Distribution (Before Balancing)')
        plt.xlabel('Steering Angle')
        plt.ylabel('Count')
        plt.legend()

    # Keep only up to max_samples per bin
    keep_indices = []
    for i in range(num_bins):
        in_bin = np.where((steering >= bin_edges[i]) &
                          (steering < bin_edges[i + 1]))[0]
        np.random.shuffle(in_bin)
        keep_indices.extend(in_bin[:max_samples].tolist())

    balanced_paths = [image_paths[i] for i in keep_indices]
    balanced_steering = steering[keep_indices]

    if visualize:
        hist2, _ = np.histogram(balanced_steering, bins=num_bins)
        plt.subplot(1, 2, 2)
        plt.bar(bin_edges[:-1], hist2,
                width=(bin_edges[1] - bin_edges[0]),
                align='edge', color='seagreen', edgecolor='black')
        plt.axhline(max_samples, color='red', linestyle='--',
                    label=f'Threshold = {max_samples}')
        plt.title('Steering Angle Distribution (After Balancing)')
        plt.xlabel('Steering Angle')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig('steering_distribution.png', dpi=150)
        plt.show()

    print(f'[INFO] Dataset balanced: {len(balanced_paths)} samples kept '
          f'(was {len(image_paths)})')
    return balanced_paths, balanced_steering


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / validation split
# ─────────────────────────────────────────────────────────────────────────────
def split_data(image_paths: list, steering: np.ndarray,
               test_size: float = 0.2, random_state: int = 42):
    """
    Split into training and validation sets.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, steering,
        test_size=test_size,
        random_state=random_state
    )
    print(f'[INFO] Train: {len(X_train)} | Validation: {len(X_val)}')
    return X_train, X_val, y_train, y_val


# ─────────────────────────────────────────────────────────────────────────────
# 4. Image preprocessing (applied at inference time too)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Apply the preprocessing pipeline required by the Nvidia model:
      1. Crop sky and hood from the image
      2. Convert BGR → YUV colour space
      3. Apply Gaussian blur to reduce noise
      4. Resize to 200 x 66 (Nvidia input size)
      5. Normalize pixel values to [0, 1]

    Parameters
    ----------
    img : np.ndarray  BGR image as returned by cv2.imread / cv2.imdecode

    Returns
    -------
    np.ndarray  shape (66, 200, 3)  float32
    """
    # Step 1 – crop: remove top 60 rows (sky) and bottom 25 rows (hood)
    img = img[60:135, :, :]

    # Step 2 – YUV colour space (as used in Nvidia paper)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Step 3 – Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 4 – Resize to Nvidia input dimensions
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Step 5 – Normalize
    img = img / 255.0

    return img.astype(np.float32)
