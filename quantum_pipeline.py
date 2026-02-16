import json
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

# ===============================
# READ CONFIG FROM R PIPELINE
# ===============================
with open("quantum_config.json", "r") as f:
    config = json.load(f)

N_FEATURES = config["num_features"]
FEATURE_NAMES = config["feature_names"]

print("Loaded config:")
print("Number of features / qubits:", N_FEATURES)
print("Selected features:", FEATURE_NAMES)

# ===============================
# BUILD FILENAMES DYNAMICALLY
# ===============================
X_TRAIN_FILE = f"X_train_quantum_{N_FEATURES}features.csv"
X_TEST_FILE  = f"X_test_quantum_{N_FEATURES}features.csv"
Y_TRAIN_FILE = f"y_train_quantum_{N_FEATURES}features.csv"
Y_TEST_FILE  = f"y_test_quantum_{N_FEATURES}features.csv"

# ===============================
# LOAD DATA FROM R PIPELINE
# ===============================
X_train = pd.read_csv(X_TRAIN_FILE).values
X_test  = pd.read_csv(X_TEST_FILE).values

y_train = pd.read_csv(Y_TRAIN_FILE)["label"].values
y_test  = pd.read_csv(Y_TEST_FILE)["label"].values

print("Data loaded successfully")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ===============================
# SCALE FEATURES FOR ANGLE ENCODING
# ===============================
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ===============================
# QUANTUM SETUP
# ===============================
N_QUBITS = N_FEATURES
N_LAYERS = 2
EPOCHS = 40
LR = 0.1

dev = qml.device("lightning.qubit", wires=N_QUBITS)

# ===============================
# QUANTUM CIRCUIT
# ===============================
@qml.qnode(dev)
def circuit(x, weights):
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)

    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    return qml.expval(qml.PauliZ(0))


def model(x, weights):
    return circuit(x, weights)

# ===============================
# LOSS FUNCTION
# ===============================
def loss(weights, X, y):
    preds = np.array([model(x, weights) for x in X])
    probs = (preds + 1) / 2
    return -np.mean(
        y * np.log(probs + 1e-6) +
        (1 - y) * np.log(1 - probs + 1e-6)
    )

# ===============================
# INITIALIZE
# ===============================
weights = np.random.normal(
    0, 0.1,
    size=(N_LAYERS, N_QUBITS, 3),
    requires_grad=True
)

opt = qml.AdamOptimizer(LR)

# ===============================
# TRAIN
# ===============================
print("Starting training...")

for epoch in range(EPOCHS):
    weights = opt.step(lambda w: loss(w, X_train, y_train), weights)

    if epoch % 10 == 0:
        current_loss = loss(weights, X_train, y_train)
        print(f"Epoch {epoch}: Loss = {current_loss:.4f}")

print("Training complete")

# ===============================
# EVALUATE
# ===============================
print("Evaluating model...")

test_preds = np.array([model(x, weights) for x in X_test])

# Convert quantum output (-1 to 1) → probability (0 to 1)
test_probs = (test_preds + 1) / 2

# ROC-AUC
auc = roc_auc_score(y_test, test_probs)
print("Quantum ROC AUC:", auc)

# ===============================
# SAVE RESULTS FOR SHINY
# ===============================
results = pd.DataFrame({
    "y_true": y_test,
    "y_prob": test_probs
})

results.to_csv("quantum_results.csv", index=False)
print("quantum_results.csv written")

