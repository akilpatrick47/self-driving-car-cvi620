import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# folder structure
DATA_DIR = 'data'
IMG_HEIGHT = 66
IMG_WIDTH = 200
NUM_BINS = 25
MAX_SAMPLES_PER_BIN = 400


def load_data(data_dir=DATA_DIR):
    csv_path = os.path.join(data_dir, 'driving_log.csv')
    df = pd.read_csv(csv_path, header=None,
                     names=['center', 'left', 'right',
                            'steering', 'throttle', 'brake', 'speed'])

    # fix paths so they work on any machine
    image_paths = []
    for path in df['center']:
        filename = os.path.basename(path.strip())
        image_paths.append(os.path.join(data_dir, 'IMG', filename))

    steering = df['steering'].values.astype(np.float32)
    print(f'[INFO] Loaded {len(image_paths)} samples from {csv_path}')
    return image_paths, steering


def balance_dataset(image_paths, steering, num_bins=NUM_BINS,
                    max_samples=MAX_SAMPLES_PER_BIN, visualize=True):
    # the simulator records mostly straight driving so we need to even it out
    hist, bin_edges = np.histogram(steering, bins=num_bins)

    if visualize:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]),
                align='edge', color='steelblue', edgecolor='black')
        plt.axhline(max_samples, color='red', linestyle='--',
                    label=f'Threshold = {max_samples}')
        plt.title('Before Balancing')
        plt.xlabel('Steering Angle')
        plt.ylabel('Count')
        plt.legend()

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
        plt.bar(bin_edges[:-1], hist2, width=(bin_edges[1] - bin_edges[0]),
                align='edge', color='seagreen', edgecolor='black')
        plt.axhline(max_samples, color='red', linestyle='--',
                    label=f'Threshold = {max_samples}')
        plt.title('After Balancing')
        plt.xlabel('Steering Angle')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig('steering_distribution.png', dpi=150)
        plt.show()

    print(f'[INFO] Dataset balanced: {len(balanced_paths)} samples kept (was {len(image_paths)})')
    return balanced_paths, balanced_steering


def split_data(image_paths, steering, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, steering,
        test_size=test_size,
        random_state=random_state
    )
    print(f'[INFO] Train: {len(X_train)} | Validation: {len(X_val)}')
    return X_train, X_val, y_train, y_val


def preprocess_image(img):
    # crop out sky and car hood, convert to YUV, blur, resize, normalize
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img.astype(np.float32)
