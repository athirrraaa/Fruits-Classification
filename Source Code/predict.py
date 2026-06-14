from joblib import load
from feature_extraction import extract_features
import os

# ==========================================
# LOAD MODEL
# ==========================================

model = load(
    "../models/fruit_svm.joblib"
)

print("Current Working Directory:")
print(os.getcwd())

# ==========================================
# INPUT IMAGE
# ==========================================

image_path = input(
    "Enter image path: ").strip().strip("'").strip("'")

print("File exists:", os.path.exists(image_path))

# ==========================================
# FEATURE EXTRACTION
# ==========================================

features = extract_features(
    image_path
)

if features is None:
    print("Image cannot be loaded.")
    exit()

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    [features]
)

print(
    "\nPredicted Fruit:",
    prediction[0]
)