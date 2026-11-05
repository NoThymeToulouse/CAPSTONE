
import numpy as np

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score
)
from test_rough_set import rough_reduct

# ======================================
# SETTINGS
# ======================================

USE_ROUGH_SET = True
ROUGH_BINS = 3

HYPO_THRESHOLD = 3.9
HIDDEN_NEURONS = 48
REGULARIZATION = 1.0
RANDOM_SEED = 42

# MATLAB columns 3, 4, 6, 7:
# HR, QTc, change HR, change QTc

FEATURE_COLUMNS = [2, 3, 5, 6]


# ======================================
# LOAD DATA
# ======================================

def load_dataset(filename):

    data = loadmat(filename)["as"]

    X = data[:, FEATURE_COLUMNS]

    glucose = data[:, 1]

    # Hypoglycaemia = -1
    # Normal = +1

    y = np.where(
        glucose < HYPO_THRESHOLD,
        -1,
        +1
    )

    return X, y


X_train, y_train = load_dataset("pat3s.mat")

X_test, y_test = load_dataset("pat4s.mat")

print("Training records:", len(y_train))
print("Testing records:", len(y_test))

print(
    "Training hypo records:",
    np.sum(y_train == -1)
)

print(
    "Testing hypo records:",
    np.sum(y_test == -1)
)

# ======================================
# ROUGH SET
# ======================================

FEATURE_NAMES = [
    "HR",
    "QTc",
    "Change HR",
    "Change QTc"
]

if USE_ROUGH_SET:

    selected_features = rough_reduct(
        X_train,
        y_train,
        n_bins=ROUGH_BINS
    )

    print("\nRough Set selected features:")


# ======================================
# PREPROCESSING
# ======================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================
# EXTREME LEARNING MACHINE
# ======================================

rng = np.random.default_rng(RANDOM_SEED)

n_features = X_train.shape[1]

W = rng.normal(
    size=(n_features, HIDDEN_NEURONS)
)

b = rng.normal(
    size=HIDDEN_NEURONS
)

# Hidden-layer outputs

H_train = np.tanh(X_train @ W + b)

# Ridge-stabilised output weights

identity = np.eye(HIDDEN_NEURONS)

beta = np.linalg.solve(
    H_train.T @ H_train
    + REGULARIZATION * identity,
    H_train.T @ y_train
)


# ======================================
# TEST MODEL
# ======================================

H_test = np.tanh(X_test @ W + b)

scores = H_test @ beta

THRESHOLD = 0.4

y_pred = np.where(
    scores < THRESHOLD,
    -1,
    +1
)


# ======================================
# MODEL EVALUATION
# ======================================

# Class order:
# +1 = non-hypoglycaemic
# -1 = hypoglycaemic

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[+1, -1]
).ravel()

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=-1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=-1,
    zero_division=0
)

print("\n===== RESULTS =====")

print("Sensitivity:", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion matrix:")
print("TN:", tn, "FP:", fp)
print("FN:", fn, "TP:", tp)