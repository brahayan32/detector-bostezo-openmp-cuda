import os
import cv2
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "processed"
)

CSV_DIR = os.path.join(
    BASE_DIR,
    "csv"
)

os.makedirs(
    CSV_DIR,
    exist_ok=True
)

splits = ["train", "val", "test"]

label_map = {
    "normal": 0,
    "yawn": 1
}

for split in splits:

    data = []

    for cls in ["normal", "yawn"]:

        folder = os.path.join(
            PROCESSED_DIR,
            split,
            cls
        )

        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        for img_name in images:

            img_path = os.path.join(
                folder,
                img_name
            )

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            img = img.astype(
                "float32"
            ) / 255.0

            vector = img.flatten()

            row = vector.tolist()

            row.append(
                label_map[cls]
            )

            data.append(
                row
            )

    df = pd.DataFrame(data)

    csv_path = os.path.join(
        CSV_DIR,
        f"{split}.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        header=False
    )

    print(
        f"{split}.csv generado"
    )

print("\nCSV generados correctamente.")