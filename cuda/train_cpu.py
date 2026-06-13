import os
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

# =====================================
# RUTAS
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_DIR = os.path.join(
    BASE_DIR,
    "csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# =====================================
# CARGAR DATASET
# =====================================

train_df = pd.read_csv(
    os.path.join(CSV_DIR, "train.csv"),
    header=None
)

val_df = pd.read_csv(
    os.path.join(CSV_DIR, "val.csv"),
    header=None
)

# =====================================
# X e Y
# =====================================

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_val = val_df.iloc[:, :-1].values
y_val = val_df.iloc[:, -1].values

# =====================================
# MODELO
# =====================================

model = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=42
)

print("\nEntrenando modelo...\n")

model.fit(
    X_train,
    y_train
)

# =====================================
# PREDICCIÓN
# =====================================

y_pred = model.predict(
    X_val
)

acc = accuracy_score(
    y_val,
    y_pred
)

print(
    f"\nAccuracy: {acc:.4f}"
)

# =====================================
# MATRIZ CONFUSIÓN
# =====================================

cm = confusion_matrix(
    y_val,
    y_pred
)

print("\nMatriz de Confusión:\n")
print(cm)

# =====================================
# REPORTE
# =====================================

report = classification_report(
    y_val,
    y_pred
)

print("\nReporte:\n")
print(report)

# =====================================
# GUARDAR PESOS
# =====================================

np.savez(
    os.path.join(
        RESULTS_DIR,
        "weights_cpu.npz"
    ),
    W1=model.coefs_[0],
    W2=model.coefs_[1],
    B1=model.intercepts_[0],
    B2=model.intercepts_[1]
)

# =====================================
# CURVA LOSS
# =====================================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    model.loss_curve_
)

plt.title(
    "Loss Curve"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.grid(True)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    )
)

plt.close()

# =====================================
# METRICS TXT
# =====================================

with open(
    os.path.join(
        RESULTS_DIR,
        "metrics.txt"
    ),
    "w"
) as f:

    f.write(
        f"Accuracy: {acc:.4f}\n\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\n"
    )

    f.write(
        report
    )

print(
    "\nResultados guardados en /results"
)