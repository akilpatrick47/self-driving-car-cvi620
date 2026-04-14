import os
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.preprocess import load_data, balance_dataset, split_data, preprocess_image, IMG_HEIGHT, IMG_WIDTH
from src.augment import augment_image

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MODEL_PATH = 'model.h5'


def batch_generator(image_paths, steering_angles, batch_size, is_training):
    num_samples = len(image_paths)
    indices = np.arange(num_samples)

    while True:
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

            yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)


def build_model():
    # Nvidia end-to-end self-driving CNN architecture
    model = Sequential([
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu',
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),

        layers.Flatten(),

        layers.Dense(100, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)  # output: steering angle
    ])

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model


def plot_history(H):
    plt.figure(figsize=(8, 4))
    plt.plot(H.history['loss'], label='Train Loss', color='steelblue')
    plt.plot(H.history['val_loss'], label='Val Loss', color='tomato')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.show()
    print('[INFO] Loss plot saved to training_loss.png')


def train():
    image_paths, steering = load_data()
    image_paths, steering = balance_dataset(image_paths, steering, visualize=True)

    X_train, X_val, y_train, y_val = split_data(image_paths, steering)

    model = build_model()
    print(model.summary())

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    steps_train = max(1, len(X_train) // BATCH_SIZE)
    steps_val = max(1, len(X_val) // BATCH_SIZE)

    print(f'[INFO] Starting training | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}')

    H = model.fit(
        batch_generator(X_train, y_train, BATCH_SIZE, is_training=True),
        steps_per_epoch=steps_train,
        validation_data=batch_generator(X_val, y_val, BATCH_SIZE, is_training=False),
        validation_steps=steps_val,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    plot_history(H)
    print(f'[INFO] Best model saved to {MODEL_PATH}')


if __name__ == '__main__':
    train()
