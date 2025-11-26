import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

FILE_LOC = "src/DatasetSorted.xlsx"

if os.path.exists(FILE_LOC):
    print("Dataset exists")

# One scaler for the input feature (avg voltage)
SCALER_X = StandardScaler()

# Model 1: avg voltage -> SOE
MODEL_SOE = LinearRegression()

# Model 2: SOE -> SOH
MODEL_SOH = LinearRegression()


# ---------------- Data Functions ----------------
def fetch_dataset():
    dataset = pd.read_excel(FILE_LOC)
    dataset = dataset.to_numpy()
    return dataset

def get_features_targets(dataset):
    """
    Feature: average of U1..U20 for each row.
    Target1: SOE (for first regression).
    Target2: SOH (for second regression).
    """
    voltages = dataset[:, 8:28]              # U1..U20
    avg_voltage = voltages.mean(axis=1)      # shape (n,)
    X = avg_voltage.reshape(-1, 1)           # shape (n, 1) for sklearn

    SOE = dataset[:, 7]                      # SOE column
    SOH = dataset[:, 29]                     # SOH column

    return X, SOE, SOH

def eval_and_print(tag, y_true, y_pred, low_thresh=None):
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"\n[{tag}] Global:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R-squared: {r2:.6f}")

    if low_thresh is not None:
        low_idx = y_true <= low_thresh
        if np.any(low_idx):
            y_true_low = y_true[low_idx]
            y_pred_low = y_pred[low_idx]
            mae_low = mean_absolute_error(y_true_low, y_pred_low)
            mse_low = mean_squared_error(y_true_low, y_pred_low)
            r2_low  = r2_score(y_true_low, y_pred_low)
            p95_abs_err_low = np.percentile(np.abs(y_true_low - y_pred_low), 95)

            print(f"\n[{tag}] Low-SOH (≤ {low_thresh}) subset:")
            print(f"Count: {low_idx.sum()} / {len(y_true)}")
            print(f"MAE_low: {mae_low:.6f}")
            print(f"MSE_low: {mse_low:.6f}")
            print(f"R2_low:  {r2_low:.3f}")
            print(f"P95 |y-ŷ| (low): {p95_abs_err_low:.6f}")
        else:
            print(f"\n[{tag}] No test samples with SOH ≤ {low_thresh}.")


# ---------------- Training the Models ----------------
dataset = fetch_dataset()
X, SOE, SOH = get_features_targets(dataset)

# Split once, keep aligned
X_train, X_test, soe_train, soe_test, soh_train, soh_test = train_test_split(
    X, SOE, SOH, test_size=0.2, random_state=42
)



# Scale input feature (avg voltage)
SCALER_X.fit(X_train)
X_train_scaled = SCALER_X.transform(X_train)
X_test_scaled  = SCALER_X.transform(X_test)

# -------- Model 1: avg voltage -> SOE --------
MODEL_SOE.fit(X_train_scaled, soe_train)

SOE_REF = np.percentile(soe_train, 95)  # high-capacity reference (approx "new" battery)
print("SOE_REF used for SOH normalization:", SOE_REF)

soe_pred_test = MODEL_SOE.predict(X_test_scaled)

eval_and_print("Model 1 (avg V -> SOE)", soe_test, soe_pred_test)

# -------- Model 2: SOE -> SOH --------
# Train second regression on true SOE vs SOH
soe_train_reshaped = soe_train.reshape(-1, 1)
soe_test_reshaped  = soe_test.reshape(-1, 1)
MODEL_SOH.fit(soe_train_reshaped, soh_train)

# Evaluate SOH prediction using TRUE SOE (upper bound performance)
soh_pred_from_true_soe = MODEL_SOH.predict(soe_test_reshaped)
eval_and_print("Model 2 (true SOE -> SOH)", soh_test, soh_pred_from_true_soe, low_thresh=0.75)

# Evaluate SOH prediction using PREDICTED SOE (realistic pipeline)
soe_pred_test_reshaped = soe_pred_test.reshape(-1, 1)
soh_pred_pipeline = MODEL_SOH.predict(soe_pred_test_reshaped)
eval_and_print("Pipeline (avg V -> SOE -> SOH)", soh_test, soh_pred_pipeline, low_thresh=0.75)

SOE_MIN = soe_train.min()
SOE_MAX = soe_train.max()
SOH_MIN = 0.5
SOH_MAX = 1.0
# ---------------- Chatbot / Helper Functions ----------------
def predict_soh_for_user_from_voltages(voltages_20):
    try:
        avg_v = float(np.mean(voltages_20))
        X_input = np.array([[avg_v]])
        X_scaled = SCALER_X.transform(X_input)

        # 1) Predict SOE
        soe_pred = MODEL_SOE.predict(X_scaled)[0]

        # 2) Normalize SOE within training range
        soe_norm = (soe_pred - SOE_MIN) / (SOE_MAX - SOE_MIN)
        soe_norm = float(np.clip(soe_norm, 0.0, 1.0))

        # 3) Map to SOH band
        soh_est = SOH_MIN + soe_norm * (SOH_MAX - SOH_MIN)

        status = "The battery is healthy." if soh_est >= 0.6 else "The battery has a problem."
        return avg_v, soe_pred, soh_est, status

    except Exception as e:
        return None, None, None, f"Could not predict SOH: {e}"


def generate_random_row(low, high):
    """Generate one sorted row of 20 voltages between low and high."""
    return sorted(np.random.uniform(low, high, 20).tolist())
# Persistent conversation memory
chat_history = []

def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        chat_history.append({"role": "user", "parts": [prompt]})
        response = model.generate_content(chat_history[-6:])
        chat_history.append({"role": "model", "parts": [response.text]})
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"


def augment_dataset(X, y, num_aug=5, noise_std=0.005):
    """
    Augment on the avg-voltage feature X (shape (n,1)) for SOE prediction.
    """
    X_aug = []
    y_aug_out = []

    for i in range(len(X)):
        for _ in range(num_aug):
            noise = np.random.normal(0, noise_std)      # scalar noise
            new_feature = X[i, 0] + noise
            new_feature = np.clip(new_feature, 3.30, 3.95)  # realistic avg V range
            X_aug.append([new_feature])
            y_aug_out.append(y[i])

    return np.array(X_aug), np.array(y_aug_out)


# ---------------- Example test calls ----------------
print("\nExample pipeline prediction using first row of dataset:")
first_voltages = dataset[0, 8:28]
print(predict_soh_for_user_from_voltages(first_voltages))

print("SOE min/max/mean:", SOE.min(), SOE.max(), SOE.mean())
print("SOH min/max/mean:", SOH.min(), SOH.max(), SOH.mean())

# Scatter plot: SOE vs avg voltage (to sanity-check correlation)
avg_v_all = X.flatten()
plt.scatter(avg_v_all, SOE, s=10)
plt.xlabel("Average Voltage")
plt.ylabel("SOE")
plt.title("SOE vs Avg Voltage")
plt.show()

# Realistic test rows – we just use their average to run the pipeline
test_rows = []

for _ in range(10):
    # each row spans a random sub-range within 3.45–3.95
    base_low = np.random.uniform(3.45, 3.80)
    base_high = base_low + np.random.uniform(0.05, 0.15)  # ensure valid spread

    row = generate_random_row(base_low, min(base_high, 3.95))
    test_rows.append(row)

print("=== 10 Random Test Rows ===")
for i, row in enumerate(test_rows, 1):
    result = predict_soh_for_user_from_voltages(row)
    print(f"Test {i} → {result}")