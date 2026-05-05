import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import random

# Load the trained digit recognition model without compiling (avoids optimizer compatibility errors)
model = load_model('digit_recognizer_improved110.h5', compile=False)

st.title("🧮 Interactive Math Tutor")

# Initialize session state for canvas refresh key
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# Helper function to generate a new math problem
def generate_problem():
    # Sum must be between 0 and 9
    ans = random.randint(0, 9)
    a = random.randint(0, ans)
    b = ans - a
    st.session_state.num1 = a
    st.session_state.num2 = b
    st.session_state.answer = ans

# Initialize math problem if not exists
if "answer" not in st.session_state:
    generate_problem()

# Handle displaying success message and balloons after a correct answer and rerun
if st.session_state.get("show_success", False):
    st.success("Correct!")
    st.balloons()
    st.session_state.show_success = False

st.subheader(f"What is {st.session_state.num1} + {st.session_state.num2}?")
st.write("✏️ Draw your answer below!")

# Canvas for digit drawing
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    width=196,
    height=196,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)

# Handle digit recognition
if canvas_result.image_data is not None:
    try:
        img = canvas_result.image_data.astype(np.uint8)

        # Convert RGBA to grayscale
        img = Image.fromarray(img).convert('L')
        img = np.array(img)

        # Resize and normalize
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = 1 - img  # Invert colors: white bg, black digits
        img_input = img.reshape(1, 28, 28, 1)

        # Only predict if the user actually drew something
        if np.count_nonzero(img > 0.1) > 10:
            prediction = model.predict(img_input, verbose=0)
            digit = str(np.argmax(prediction))
            
            # Validation logic
            if int(digit) == st.session_state.answer:
                # Set a flag to show success message and balloons on next run
                st.session_state.show_success = True
                generate_problem()
                
                # Clear canvas by changing key and rerunning
                st.session_state.canvas_key += 1
                st.rerun()
            else:
                st.error("Not quite! Try again.")

    except Exception as e:
        st.warning(f"Could not process image: {str(e)}")

# UI Controls
col1, col2 = st.columns(2)

with col1:
    # Button to skip the current problem and generate a new one
    if st.button("🔄 New Problem"):
        generate_problem()
        st.session_state.canvas_key += 1
        st.rerun()

with col2:
    # Button to just reset the canvas without changing the problem
    if st.button("🗑️ Clear Canvas"):
        st.session_state.canvas_key += 1
        st.rerun()
