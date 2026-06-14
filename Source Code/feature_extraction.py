import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ==========================================
# IMAGE SIZE
# ==========================================

IMG_SIZE = (128, 128)

# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_image(img_path):

    print("Trying to load image:")
    print(img_path)

    img = cv2.imread(img_path)

    print("Image object:", img)

    if img is None:
        print("FAILED TO LOAD IMAGE")
        return None

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # Gaussian Filter
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Median Filter
    img = cv2.medianBlur(img, 5)

    return img

# ==========================================
# COLOR FEATURE EXTRACTION
# ==========================================

def extract_color_features(img):

    hist_b = cv2.calcHist(
        [img], [0], None, [32], [0, 256]
    )

    hist_g = cv2.calcHist(
        [img], [1], None, [32], [0, 256]
    )

    hist_r = cv2.calcHist(
        [img], [2], None, [32], [0, 256]
    )

    hist = np.concatenate([
        hist_b.flatten(),
        hist_g.flatten(),
        hist_r.flatten()
    ])

    hist = hist / np.sum(hist)

    return hist

# ==========================================
# SHAPE FEATURE EXTRACTION
# ==========================================

def extract_shape_features(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return np.zeros(4)

    cnt = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(
        cnt,
        True
    )

    if perimeter == 0:
        circularity = 0
    else:
        circularity = (
            4 * np.pi * area
        ) / (perimeter ** 2)

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = w / h

    return np.array([
        area,
        perimeter,
        circularity,
        aspect_ratio
    ])

# ==========================================
# TEXTURE FEATURE EXTRACTION (GLCM)
# ==========================================

def extract_texture_features(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        'contrast'
    )[0, 0]

    correlation = graycoprops(
        glcm,
        'correlation'
    )[0, 0]

    energy = graycoprops(
        glcm,
        'energy'
    )[0, 0]

    homogeneity = graycoprops(
        glcm,
        'homogeneity'
    )[0, 0]

    return np.array([
        contrast,
        correlation,
        energy,
        homogeneity
    ])

# ==========================================
# COMBINE FEATURES
# ==========================================

def extract_features(img_path):

    img = preprocess_image(img_path)

    if img is None:
        return None

    color_features = extract_color_features(img)

    shape_features = extract_shape_features(img)

    texture_features = extract_texture_features(img)

    features = np.concatenate([
        color_features,
        shape_features,
        texture_features
    ])

    return features