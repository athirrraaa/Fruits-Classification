import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction import extract_features

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from joblib import dump

# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset(dataset_path):

    X = []
    y = []

    classes = sorted(
        os.listdir(dataset_path)
    )

    for label in classes:

        folder = os.path.join(
            dataset_path,
            label
        )

        if not os.path.isdir(folder):
            continue

        print(f"Loading {label}...")

        for filename in os.listdir(folder):

            image_path = os.path.join(
                folder,
                filename
            )

            features = extract_features(
                image_path
            )

            if features is not None:

                X.append(features)

                y.append(label)

    return np.array(X), np.array(y)

# ==========================================
# DATASET PATH
# ==========================================

TRAIN_DIR = "../Fruits Classification/train"

TEST_DIR = "../Fruits Classification/test"

# ==========================================
# LOAD TRAIN DATA
# ==========================================

print("Loading Training Dataset...")

X_train, y_train = load_dataset(
    TRAIN_DIR
)

print("Loading Testing Dataset...")

X_test, y_test = load_dataset(
    TEST_DIR
)

print()

print("Training Shape :",
      X_train.shape)

print("Testing Shape :",
      X_test.shape)

# ==========================================
# SVM MODEL
# ==========================================

print()
print("Training SVM...")

svm_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        )
    )
])

svm_model.fit(
    X_train,
    y_train
)

pred_svm = svm_model.predict(
    X_test
)

print()
print("===== SVM RESULT =====")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred_svm
    )
)

print()

print(
    classification_report(
        y_test,
        pred_svm
    )
)

# ==========================================
# CONFUSION MATRIX SVM
# ==========================================

cm_svm = confusion_matrix(
    y_test,
    pred_svm
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.unique(y_test),
    yticklabels=np.unique(y_test)
)

plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "../results/confusion_matrix_svm.png"
)

plt.close()

dump(
    svm_model,
    "../models/fruit_svm.joblib"
)

print(
    "SVM Model Saved!"
)

# ==========================================
# KNN MODEL
# ==========================================

print()
print("Training KNN...")

knn_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "knn",
        KNeighborsClassifier(
            n_neighbors=5
        )
    )
])

knn_model.fit(
    X_train,
    y_train
)

pred_knn = knn_model.predict(
    X_test
)

print()
print("===== KNN RESULT =====")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred_knn
    )
)

print()

print(
    classification_report(
        y_test,
        pred_knn
    )
)

dump(
    knn_model,
    "../models/fruit_knn.joblib"
)

print(
    "KNN Model Saved!"
)

print()
print("Training Finished!")

# ==========================================
# CONFUSION MATRIX KNN
# ==========================================

cm_knn = confusion_matrix(
    y_test,
    pred_knn
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=np.unique(y_test),
    yticklabels=np.unique(y_test)
)

plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "../results/confusion_matrix_knn.png"
)

plt.close()