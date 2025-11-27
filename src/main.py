import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

FILE_LOC = "src/Dataset.xlsx"

SCALER = StandardScaler()
MODEL = LinearRegression()

def fetch_dataset():
    dataset = pd.read_excel(FILE_LOC)
    return dataset.to_numpy()

def get_values(dataset):
    x = dataset[:, 8:28]
    y = dataset[:, 29]
    return x, y

def fit_model(X_train, y_train, sample_weight=None):
    MODEL.fit(X_train, y_train, sample_weight=sample_weight)

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

def eval_and_print(tag, y_true, y_pred, low_thresh=0.75):
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"\n[{tag}] Global:")
    print(f"MSE: {mse:.6f}")
    print(f"R2:  {r2:.6f}")

    low_idx = y_true <= low_thresh
    if np.any(low_idx):
        y_true_low = y_true[low_idx]
        y_pred_low = y_pred[low_idx]
        mae_low = mean_absolute_error(y_true_low, y_pred_low)
        mse_low = mean_squared_error(y_true_low, y_pred_low)
        r2_low  = r2_score(y_true_low, y_pred_low)
        p95_abs = np.percentile(np.abs(y_true_low - y_pred_low), 95)
        print(f"\n[{tag}] Low-SOH subset:")
        print(f"Count: {low_idx.sum()} / {len(y_true)}")
        print(f"MAE_low: {mae_low:.6f}")
        print(f"MSE_low: {mse_low:.6f}")
        print(f"R2_low:  {r2_low:.3f}")
        print(f"P95 |y-ŷ|: {p95_abs:.6f}")

eval_and_print("UNBALANCED", y_test, y_pred_base)
eval_and_print("BALANCED",   y_test, y_pred)

def predict_soh(inputs):
    """
    inputs: list of 20 voltages (3.0 to 4.2)
    returns: SOH between 0 and 1
    """
    if len(inputs) != 20:
        raise ValueError("Expected 20 voltage inputs")

    for v in inputs:
        if v < 3.0 or v > 4.2:
            raise ValueError("Voltage values must be between 3.0 and 4.2")

    avg_voltage = np.mean(inputs)

    soh = (avg_voltage - 3.0) / 1.2

    soh = float(np.clip(soh, 0.0, 1.0))

    return soh

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
