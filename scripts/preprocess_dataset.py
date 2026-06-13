import os
import cv2

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

splits = ["train", "val", "test"]
classes = ["yawn", "normal"]

for split in splits:

    for cls in classes:

        input_folder = os.path.join(
            DATASET_DIR,
            split,
            cls
        )

        output_folder = os.path.join(
            PROCESSED_DIR,
            split,
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

        print(
            f"\nProcesando {split}/{cls}"
        )

        for img_name in images:

            img_path = os.path.join(
                input_folder,
                img_name
            )

            img = cv2.imread(
                img_path
            )

            if img is None:
                continue

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

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

            blur = cv2.GaussianBlur(
                sobel,
                (5, 5),
                0
            )

            resized = cv2.resize(
                blur,
                (64, 64)
            )

            output_path = os.path.join(
                output_folder,
                img_name
            )

            cv2.imwrite(
                output_path,
                resized
            )

print("\nPreprocesamiento terminado.")