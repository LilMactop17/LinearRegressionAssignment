import streamlit as st
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
from main import MODEL, SCALER


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# Initialize chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Predicting the Battery SOH
def predict_battery_soh(inputs):
    scaled_input = SCALER.transform([inputs])
    soh_pred = MODEL.predict(scaled_input)[0]
    return soh_pred


# Talking to Gemnini
def ask_gemini(prompt):
    """Chat with Gemini using limited context (faster)"""
    try:
        st.session_state.chat_history.append({"role": "user", "parts": [prompt]})
        context = st.session_state.chat_history[-4:] if len(st.session_state.chat_history) > 4 else st.session_state.chat_history

        with st.spinner("🤖 Thinking..."):
            response = GEMINI_MODEL.generate_content(context)

        st.session_state.chat_history.append({"role": "model", "parts": [response.text]})
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# Streamlit Page Setup
st.set_page_config(page_title="Battery SOH Chatbot", page_icon="🔋", layout="centered")

# Styles
st.markdown("""
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
.stButton > button:hover {
    box-shadow: 0 0 10px rgba(108, 64, 239, 0.4);
    transform: translateY(-2px);
}

input, textarea, [data-baseweb="input"] > div {
    background-color: #ffffff !important;
    color: #333 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.chatbox {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
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
    color: #333 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 25px !important;
    padding: 0.8em 1.2em !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Fix Streamlit’s internal wrappers that create stacked borders */
[data-testid="stChatInput"] div,
[data-testid="stChatInput"] iframe,
[data-testid="stChatInput"] > div > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

=
[data-testid="stChatInput"]:focus-within,
div[role="textbox"]:focus {
    border-color: #a87fff !important;
    box-shadow: 0 0 0 3px rgba(168, 127, 255, 0.25) !important;
}

            
button[kind="secondaryFormSubmit"] {
    background: linear-gradient(135deg, #6c40ef, #a87fff);
    color: white !important;
    border-radius: 50%;
    padding: 0.5em 0.7em !important;
    margin-left: 10px;
    border: none !important;
    box-shadow: 0 0 6px rgba(108, 64, 239, 0.4);
    transition: all 0.25s ease-in-out;
}
button[kind="secondaryFormSubmit"]:hover {
    box-shadow: 0 0 10px rgba(108, 64, 239, 0.6);
    transform: scale(1.05);
}


.stSuccess, .stError {
    background-color: #ffffff !important;
    color: #333 !important;
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    padding: 0.5em 1em !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)


# Streamlit Content
st.title("Battery Health Chatbot")
st.caption("Predict SOH, view your battery’s health, and chat naturally with Gemini.")

# Battery Input
st.subheader("Enter Cell Voltage Readings")
cols = st.columns(5)
user_inputs = [cols[i % 5].number_input(f"U{i+1}", value=3.6, step=0.01, key=f"cell_{i}") for i in range(20)]

if st.button("Analyze Battery"):
    try:
        soh_pred = predict_battery_soh(user_inputs)
        avg_msg = f"Predicted SOH: {soh_pred:.2f}"
        if soh_pred < 0.6:
            health_msg = "The battery has a problem."
            prompt = "The battery SOH is low. Suggest ways to improve or manage it."
        else:
            health_msg = "The battery is healthy."
            prompt = "The battery SOH is good. Suggest maintenance tips to keep it healthy."

        st.session_state.chat_history.append({"role": "model", "parts": [f"{avg_msg}\n{health_msg}"]})
        ask_gemini(prompt)
        st.success("Battery analysis complete. You can now chat below!")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Chat Interface
st.subheader("Chat Interface")

st.markdown('<div class="chatbox">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["parts"][0]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["parts"][0]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

chat_input = st.chat_input("Type your question or message...")

if chat_input:
    ask_gemini(chat_input)
    st.rerun()

if st.button("Clear Memory"):
    st.session_state.chat_history.clear()
    st.success("Chat memory cleared! Start fresh.")
