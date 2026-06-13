import os
import random
import shutil

# ==========================
# CONFIGURACIÓN
# ==========================

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "val")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================
# CREAR CARPETAS
# ==========================

classes = ["yawn", "normal"]

for cls in classes:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

# ==========================
# DIVISIÓN
# ==========================

for cls in classes:

    source_folder = os.path.join(RAW_DIR, cls)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    random.shuffle(images)

    total = len(images)
    
    if total == 0:
        print(f"No se encontraron imágenes en {source_folder}")
        continue

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for img in train_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(VAL_DIR, cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(f"\nClase: {cls}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")

print("\nDataset dividido correctamente.")