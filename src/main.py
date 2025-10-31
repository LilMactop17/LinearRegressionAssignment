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

FILE_LOC = "Dataset.xlsx"

if os.path.exists(FILE_LOC):
    print("Dataset exists")

SCALER = StandardScaler()
MODEL = LinearRegression()  


# Data Functions
def fetch_dataset():
    dataset = pd.read_excel(FILE_LOC)
    dataset = dataset.to_numpy()
    return dataset

def get_values(dataset):
    x = dataset[:, 8:28]
    y = dataset[:, 29]
    return x, y

def scale(x):
    return SCALER.fit_transform(x)

def fit_model(X_train, y_train, sample_weight=None):
    MODEL.fit(X_train, y_train, sample_weight=sample_weight)

def eval_and_print(tag, y_true, y_pred, low_thresh=0.75):
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"\n[{tag}] Global:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R-squared: {r2:.6f}")

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


# Training the Model
dataset = fetch_dataset()
x, y = get_values(dataset)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

X_train = SCALER.fit_transform(X_train)
X_test  = SCALER.transform(X_test)

MODEL_base = LinearRegression()
MODEL_base.fit(X_train, y_train)
y_pred_base = MODEL_base.predict(X_test)

weights_train = np.where(y_train < 0.75, 4.0, 1.0)
fit_model(X_train, y_train, weights_train)
y_pred = MODEL.predict(X_test)

eval_and_print("UNBALANCED", y_test, y_pred_base, low_thresh=0.75)
eval_and_print("BALANCED",   y_test, y_pred,      low_thresh=0.75)

low_idx = y_test <= 0.75
if np.any(low_idx):
    mae_low_base = mean_absolute_error(y_test[low_idx], y_pred_base[low_idx])
    mae_low_bal  = mean_absolute_error(y_test[low_idx], y_pred[low_idx])
    mse_low_base = mean_squared_error(y_test[low_idx], y_pred_base[low_idx])
    mse_low_bal  = mean_squared_error(y_test[low_idx], y_pred[low_idx])

    if mae_low_base > 0:
        mae_impr = (mae_low_base - mae_low_bal) / mae_low_base * 100
        mse_impr = (mse_low_base - mse_low_bal) / mse_low_base * 100
        print(f"\nLow-SOH MAE improvement (Balanced vs Unbalanced): {mae_impr:.1f}%")
        print(f"Low-SOH MSE improvement (Balanced vs Unbalanced): {mse_impr:.1f}%")


# Chatbot Functions
def predict_soh_for_user():
    try:
        sample_input = np.random.rand(21)
        scaled_input = SCALER.transform([sample_input])
        soh_pred = MODEL.predict(scaled_input)[0]

        if soh_pred < 0.6:
            return f"Predicted SOH: {soh_pred:.2f} → The battery has a problem."
        else:
            return f"Predicted SOH: {soh_pred:.2f} → The battery is healthy."
    except Exception as e:
        return f"Could not predict SOH: {e}"

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

# Example test calls
print(predict_soh_for_user())


