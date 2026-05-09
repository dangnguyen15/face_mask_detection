import os
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from model import build_model
from dataset_loader import load_dataset


# =========================
# PATH
# =========================

MODEL_DIR = "../models"

MODEL_PATH = "../models/face_mask_model.keras"

DATASET_PATH = "../dataset"

GRAPH_PATH = "../models/training_graph.png"


# =========================
# CREATE MODEL FOLDER
# =========================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

train_data, val_data = load_dataset(DATASET_PATH)


# =========================
# BUILD MODEL
# =========================

print("Building model...")

model = build_model()


# =========================
# CALLBACKS
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# =========================
# TRAIN MODEL
# =========================

EPOCHS = 5

print("Training model...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)


# =========================
# SAVE FINAL MODEL
# =========================

print("Saving final model...")

model.save(MODEL_PATH)

print("Model saved successfully!")


# =========================
# TRAINING HISTORY
# =========================

train_acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

train_loss = history.history['loss']

val_loss = history.history['val_loss']

epochs_range = range(1, len(train_acc) + 1)


# =========================
# FIND BEST EPOCH
# =========================

best_epoch = val_acc.index(max(val_acc)) + 1

best_val_acc = max(val_acc)

print("\n=========================")
print(f"Best Epoch : {best_epoch}")
print(f"Best Validation Accuracy : {best_val_acc:.4f}")
print("=========================\n")


# =========================
# OVERFITTING CHECK
# =========================

final_train_acc = train_acc[-1]

final_val_acc = val_acc[-1]

gap = final_train_acc - final_val_acc

print(f"Train Accuracy : {final_train_acc:.4f}")

print(f"Validation Accuracy : {final_val_acc:.4f}")

print(f"Accuracy Gap : {gap:.4f}")


if gap > 0.10:
    print("\nWARNING: Possible Overfitting Detected!")
else:
    print("\nModel is Generalizing Well.")


# =========================
# PLOT GRAPH
# =========================

plt.figure(figsize=(12, 5))


# Accuracy Graph

plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_acc, label='Train Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Training vs Validation Accuracy')

plt.legend()


# Loss Graph

plt.subplot(1, 2, 2)

plt.plot(epochs_range, train_loss, label='Train Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Training vs Validation Loss')

plt.legend()


plt.tight_layout()

plt.savefig(GRAPH_PATH)

plt.show()


print(f"\nTraining graph saved at: {GRAPH_PATH}")