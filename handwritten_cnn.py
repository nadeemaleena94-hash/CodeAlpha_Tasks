"""
CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
CNN on MNIST (digits) with an option for EMNIST (letters).

Install: pip install tensorflow matplotlib scikit-learn
         (EMNIST mode also needs: pip install tensorflow-datasets)
Run:     python handwritten_cnn.py              # MNIST digits (~99% accuracy)
         python handwritten_cnn.py --emnist     # EMNIST letters (26 classes)
"""
import argparse
import keras
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ap = argparse.ArgumentParser()
ap.add_argument("--emnist", action="store_true")
ap.add_argument("--epochs", type=int, default=15)
args = ap.parse_args()


def load_data():
    if not args.emnist:
        import keras
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    return xtr, ytr, xte, yte, [str(i) for i in range(10)]
    import tensorflow_datasets as tfds
    tr, te = tfds.as_numpy(tfds.load("emnist/letters", split=["train", "test"], as_supervised=True))
    xtr = np.array([np.transpose(i[0].squeeze()) for i in tr])   # EMNIST images are transposed
    ytr = np.array([i[1] for i in tr]) - 1                       # labels 1..26 -> 0..25
    xte = np.array([np.transpose(i[0].squeeze()) for i in te])
    yte = np.array([i[1] for i in te]) - 1
    return xtr, ytr, xte, yte, [chr(65 + i) for i in range(26)]


xtr, ytr, xte, yte, classes = load_data()
n_classes = len(classes)

# Preprocess: scale to [0,1], add channel dimension
xtr = (xtr.astype("float32") / 255.0)[..., None]
xte = (xte.astype("float32") / 255.0)[..., None]
print("Train:", xtr.shape, "Test:", xte.shape, "Classes:", n_classes)

# Model ------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.RandomRotation(0.05),          # light augmentation
    layers.RandomZoom(0.1),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

cb = [callbacks.EarlyStopping(patience=3, restore_best_weights=True),
      callbacks.ReduceLROnPlateau(patience=2, factor=0.5)]
hist = model.fit(xtr, ytr, validation_split=0.1, epochs=args.epochs, batch_size=128, callbacks=cb)

# Evaluate -----------------------------------------------------------------
loss, acc = model.evaluate(xte, yte, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")
pred = model.predict(xte).argmax(axis=1)
print(classification_report(yte, pred, target_names=classes))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(hist.history["accuracy"], label="train")
ax[0].plot(hist.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].legend()
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=classes).plot(
    ax=ax[1], colorbar=False, xticks_rotation=90)
ax[1].set_title("Confusion Matrix")
plt.tight_layout(); plt.savefig("handwritten_results.png", dpi=150)

# Sample predictions
idx = np.random.choice(len(xte), 12, replace=False)
plt.figure(figsize=(10, 4))
for k, i in enumerate(idx):
    plt.subplot(2, 6, k + 1)
    plt.imshow(xte[i].squeeze(), cmap="gray"); plt.axis("off")
    plt.title(f"P:{classes[pred[i]]} T:{classes[yte[i]]}", fontsize=8)
plt.tight_layout(); plt.savefig("sample_predictions.png", dpi=150)

model.save("handwritten_cnn.keras")
print("Saved model -> handwritten_cnn.keras")