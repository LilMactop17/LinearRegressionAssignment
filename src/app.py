import numpy as np
import google.generativeai as genai
import os
import re
import streamlit as st
from dotenv import load_dotenv
from main import predict_soh, SCALER, MODEL   

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")


# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None

if "soh_display" not in st.session_state:
    st.session_state.soh_display = None


# Gemini Chat Function (Stable)
def ask_gemini(prompt):
    try:
        context = st.session_state.chat_history[-6:]  # short memory
        messages = [{"role": role, "parts": [msg]} for role, msg in context]
        messages.append({"role": "user", "parts": [prompt]})
        response = GEMINI_MODEL.generate_content(messages)
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"


# Streamlit UI
st.title("Battery SOH Analyzer & Chatbot")
st.write("Enter 20 voltage values (0–1 scale).")


cols = st.columns(5)
u_values = []

labels = [f"U{i}" for i in range(1, 21)]
for idx, label in enumerate(labels):
    col = cols[idx % 5]
    value = col.number_input(
        label,
        min_value=3.00,
        max_value=4.20,
        value=3.60,
        step=0.01,
        format="%.2f"
    )
    u_values.append(value)


# Analyze Battery Button
if st.button("Analyze Battery"):
    try:
        soh = predict_soh(u_values)
        st.session_state.soh_display = soh
    except Exception as e:
        st.error(f"Prediction error: {e}")



# Display SOH Result
if st.session_state.soh_display is not None:
    soh = st.session_state.soh_display

    st.markdown("## 🔍 Analysis Result")
    st.success(f"SOH: {soh:.2f}")

    if soh < 0.75:
        st.warning("The battery is in critical condition.")
    elif soh < 0.85:
        st.info("The battery is moderately worn.")
    else:
        st.success("The battery is healthy.")


# Chat Interface
st.markdown("## Chat Interface")

prompt = st.text_input("Ask a question...")

if st.button("Send") and prompt.strip():

    if prompt != st.session_state.last_prompt:

        if st.session_state.soh_display is not None and any(
            k in prompt.lower() for k in ["soh", "health", "battery", "condition"]
        ):
            soh_val = st.session_state.soh_display
            full_prompt = f"The battery SOH is {soh_val:.2f}. {prompt}"
        else:
            full_prompt = prompt

        reply = ask_gemini(full_prompt)

        st.session_state.chat_history.append(("user", prompt))
        st.session_state.chat_history.append(("assistant", reply))

        st.session_state.last_prompt = prompt

# Display Chat History
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"** You:** {msg}")
    else:
        st.markdown(f"** Assistant:** {msg}")


# Clear Chat Memory
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.last_prompt = None
    st.success("Chat cleared.")

