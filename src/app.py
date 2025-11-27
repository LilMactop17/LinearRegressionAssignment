import streamlit as st
import numpy as np
import google.generativeai as genai
import os
import re
from dotenv import load_dotenv
from main import predict_soh_for_user_from_voltages

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "soh_threshold" not in st.session_state:
    st.session_state.soh_threshold = 0.6

def analyze_battery_soh(voltages_20):
    avg_v, soe_pred, soh_est, status = predict_soh_for_user_from_voltages(voltages_20)
    if avg_v is None:
        raise ValueError(status)
    return avg_v, soe_pred, soh_est, status

def ask_gemini(prompt):
    try:
        context = st.session_state.chat_history[-4:] if len(st.session_state.chat_history) > 4 else st.session_state.chat_history
        with st.spinner("Thinking..."):
            response = GEMINI_MODEL.generate_content(context)
        bot_text = response.text if hasattr(response, "text") else str(response)
        st.session_state.chat_history.append({"role": "model", "parts": [bot_text]})
        return bot_text
    except Exception as e:
        err_msg = f"Gemini error: {e}"
        st.session_state.chat_history.append({"role": "model", "parts": [err_msg]})
        return err_msg

def maybe_update_threshold_from_text(text):
    lower = text.lower()
    if "threshold" not in lower:
        return False
    match = re.search(r"(\d+(\.\d+)?)", lower)
    if not match:
        return False
    try:
        val = float(match.group(1))
    except ValueError:
        return False
    if 0.0 <= val <= 1.0:
        st.session_state.soh_threshold = val
        st.session_state.chat_history.append({"role": "model", "parts": [f"SOH unhealthy threshold updated to {val:.2f}."]})
        return True
    st.session_state.chat_history.append({"role": "model", "parts": ["Threshold must be between 0.0 and 1.0."]})
    return True

def parse_pasted_voltages(raw_text):
    if not raw_text.strip():
        return None, None
    cleaned = raw_text.replace("[", "").replace("]", "")
    parts = cleaned.replace(",", " ").split()
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None, "Could not parse voltages."
    if len(vals) != 20:
        return None, f"Expected 20 values, but got {len(vals)}."
    return vals, None

st.set_page_config(page_title="Battery SOH Chatbot", page_icon=None, layout="centered")

st.markdown(
    """
<style>

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], [class*="block-container"], [data-testid="stVerticalBlock"] {
    background-color: #f7f9fc !important;
    color: #2a2a2a !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #5b18d1 !important;
    text-shadow: 0 0 4px rgba(91, 24, 209, 0.2);
}

label, p, span, div {
    color: #333 !important;
}

[data-testid="stSidebar"], [data-testid="stHeader"] {
    background-color: #f7f9fc !important;
}

.stButton > button {
    background: linear-gradient(90deg, #6c40ef, #a87fff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    transition: all 0.3s ease;
}

input, textarea, [data-baseweb="input"] > div {
    background-color: #ffffff !important;
    color: #333 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
}

.chatbox {
    background-color: #ffffff;
    max-height: 600px;
    overflow-y: auto;
    box-shadow: 0 0 8px rgba(0,0,0,0.05);
}

.user-bubble {
    background-color: #6c40ef;
    color: #fff;
    border-radius: 12px 12px 0 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
    text-align: right;
    max-width: 80%;
    float: right;
    clear: both;
}

.bot-bubble {
    background-color: #f1f0f6;
    color: #333;
    border-radius: 12px 12px 12px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    text-align: left;
    max-width: 80%;
    float: left;
    clear: both;
}

[data-testid="stChatInput"] {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

[data-testid="stChatInput"] textarea,
[data-baseweb="textarea"] textarea,
div[role="textbox"],
[data-testid="stChatInputTextArea"] {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 25px !important;
    padding: 0.8em 1.2em !important;
}

[data-testid="stChatInput"] div,
[data-testid="stChatInput"] iframe,
[data-testid="stChatInput"] > div > div {
    border: none !important;
    background: transparent !important;
}

</style>
""",
    unsafe_allow_html=True,
)

st.title("Battery Health Chatbot")
st.caption("Predict SOH, view your battery’s health, and chat naturally with Gemini.")

st.sidebar.subheader("Settings")
threshold = st.sidebar.slider(
    "SOH threshold for 'unhealthy' battery",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state.soh_threshold),
    step=0.01,
)
st.session_state.soh_threshold = threshold
st.sidebar.write(f"Current threshold: {threshold:.2f}")

DEFAULT_VALUES = [
    3.9051, 3.8924, 3.8813, 3.8749, 3.8867,
    3.9012, 3.9178, 3.9034, 3.8895, 3.8799,
    3.8654, 3.8731, 3.8872, 3.9026, 3.9153,
    3.8987, 3.8841, 3.8729, 3.8614, 3.8785,
]

st.subheader("Enter Cell Voltage Readings")

cols = st.columns(5)
user_inputs = [
    cols[i % 5].number_input(
        f"U{i+1}", value=float(DEFAULT_VALUES[i]), step=0.01, key=f"cell_{i}"
    )
    for i in range(20)
]

raw_array = st.text_area(
    "Or paste a list / CSV of 20 voltages (optional):",
    placeholder="e.g. 3.90 3.88 3.87 ...",
    key="pasted_voltages",
)

if st.button("Analyze Battery"):
    try:
        if raw_array.strip():
            parsed, err = parse_pasted_voltages(raw_array)
            if err:
                st.error(err)
                st.stop()
            voltages = parsed
        else:
            voltages = user_inputs

        avg_v, soe_pred, soh_est, status = analyze_battery_soh(voltages)

        msg = (
            f"Battery Analysis Result\n\n"
            f"Average voltage: {avg_v:.4f} V\n"
            f"Estimated SOE: {soe_pred:.3f}\n"
            f"Estimated SOH: {soh_est:.3f}\n"
            f"Status: {status}\n"
            f"Threshold used: {st.session_state.soh_threshold:.2f}"
        )

        st.session_state.chat_history.append({"role": "model", "parts": [msg]})

        if soh_est < st.session_state.soh_threshold:
            prompt = (
                f"The battery SOH is {soh_est:.2f}, below the threshold of {st.session_state.soh_threshold:.2f}. "
                "Provide suggestions to manage a degraded battery."
            )
        else:
            prompt = (
                f"The battery SOH is {soh_est:.2f}, above the threshold of {st.session_state.soh_threshold:.2f}. "
                "Provide tips to maintain good battery health."
            )

        ask_gemini(prompt)
        st.success("Battery analysis complete. You can now chat below.")

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.subheader("Chat Interface")
st.markdown(
    "You can type things like `set threshold to 0.7` to change the threshold."
)

st.markdown('<div class="chatbox">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["parts"][0]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["parts"][0]}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

chat_input = st.chat_input("Type your question or message...")

if chat_input:
    user_msg = chat_input.strip()
    st.session_state.chat_history.append({"role": "user", "parts": [user_msg]})
    handled = maybe_update_threshold_from_text(user_msg)
    if not handled:
        ask_gemini(user_msg)
    st.rerun()

if st.button("Clear Memory"):
    st.session_state.chat_history.clear()
    st.success("Chat memory cleared.")
