import os
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.preprocess import (load_data, balance_dataset, split_data,
                        preprocess_image, IMG_HEIGHT, IMG_WIDTH)
from src.augment import augment_image


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 64
LEARNING_RATE = 1e-4
MODEL_PATH  = 'model.h5'   # saved to project root (matches TestSimulation.py)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Generator
#   - Training  : loads image → augment → preprocess
#   - Validation: loads image → preprocess only
# ─────────────────────────────────────────────────────────────────────────────
def batch_generator(image_paths: list, steering_angles: np.ndarray,
                    batch_size: int, is_training: bool):
    """
    Infinite generator that yields (X_batch, y_batch) tuples.

    Parameters
    ----------
    image_paths    : list of file paths to center-camera images
    steering_angles: matching steering values
    batch_size     : number of samples per batch
    is_training    : if True, apply data augmentation
    """
    num_samples = len(image_paths)
    indices = np.arange(num_samples)

    while True:   # Keras expects an infinite generator
        np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start:start + batch_size]

            X_batch = []
            y_batch = []

            for i in batch_idx:
                img = cv2.imread(image_paths[i])
                angle = steering_angles[i]

                if is_training:
                    img, angle = augment_image(img, angle)

                img = preprocess_image(img)

                X_batch.append(img)
                y_batch.append(angle)

            yield np.array(X_batch, dtype=np.float32), \
                  np.array(y_batch,  dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Nvidia CNN Architecture
#   Reference: "End to End Learning for Self-Driving Cars" (Bojarski et al.)
# ─────────────────────────────────────────────────────────────────────────────
def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)) -> Sequential:
    """
    Build and compile the Nvidia self-driving CNN.

    Architecture
    ------------
    Input  → 5 Conv layers → Flatten → 4 Dense layers → Steering output (1)

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential([
        # Convolutional feature extractor
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu',
                      input_shape=input_shape),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),

        layers.Flatten(),

        # Fully connected head
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='relu'),

        # Single steering angle output (regression)
        layers.Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse'      # Mean Squared Error for regression
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def plot_history(H, epochs: int):
    """Save and display training / validation loss curves."""
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(H.history['loss']) + 1),
             H.history['loss'], label='Train Loss', color='steelblue')
    plt.plot(range(1, len(H.history['val_loss']) + 1),
             H.history['val_loss'], label='Val Loss', color='tomato')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training / Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.show()
    print('[INFO] Loss plot saved to training_loss.png')


def train():

    # 1. Load + balance data
    image_paths, steering = load_data()
    image_paths, steering = balance_dataset(image_paths, steering,
                                             visualize=True)

    # 2. Train / validation split
    X_train, X_val, y_train, y_val = split_data(image_paths, steering)

    # 3. Build model
    model = build_model()
    print(model.summary())

    # 4. Set up callbacks
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_loss',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1)
    ]

    # 5. Train using generators
    steps_train = max(1, len(X_train) // BATCH_SIZE)
    steps_val   = max(1, len(X_val)   // BATCH_SIZE)

    print(f'[INFO] Starting training | Epochs: {EPOCHS} | '
          f'Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}')

    H = model.fit(
        batch_generator(X_train, y_train, BATCH_SIZE, is_training=True),
        steps_per_epoch=steps_train,
        validation_data=batch_generator(X_val, y_val, BATCH_SIZE,
                                         is_training=False),
        validation_steps=steps_val,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 6. Plot and save loss curves
    plot_history(H, EPOCHS)

    print(f'[INFO] Best model saved to {MODEL_PATH}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train()
