import os
import cv2

# =====================================
# RUTAS
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

classes = ["yawn", "normal"]

# =====================================
# PROCESAMIENTO
# =====================================

for cls in classes:

    input_folder = os.path.join(
        TRAIN_DIR,
        cls
    )

    output_folder = os.path.join(
        OUTPUT_DIR,
        cls
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    images = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(f"\nProcesando clase: {cls}")

    for img_name in images:

        img_path = os.path.join(
            input_folder,
            img_name
        )

        img = cv2.imread(img_path)

        if img is None:
            continue

        # --------------------------
        # Escala de grises
        # --------------------------

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # --------------------------
        # Sobel
        # --------------------------

        sobelx = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobely = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        sobel = cv2.magnitude(
            sobelx,
            sobely
        )

        sobel = cv2.convertScaleAbs(
            sobel
        )

        # --------------------------
        # Gaussiano
        # --------------------------

        blur = cv2.GaussianBlur(
            sobel,
            (5, 5),
            0
        )

        # --------------------------
        # Resize 64x64
        # --------------------------

        resized = cv2.resize(
            blur,
            (64, 64)
        )

        # --------------------------
        # Guardar
        # --------------------------

        output_path = os.path.join(
            output_folder,
            img_name
        )

        cv2.imwrite(
            output_path,
            resized
        )

    print(
        f"{len(images)} imágenes procesadas"
    )

print("\nProcesamiento completado.")