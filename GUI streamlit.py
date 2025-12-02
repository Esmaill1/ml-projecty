import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained digit recognition model without compiling (avoids optimizer compatibility errors)
model = load_model('digit_recognizer_improved110.h5', compile=False)

st.title("🖋️ Handwriting Typer: Draw a digit, it types it, and clears!")

# Initialize session state for typed digits and canvas refresh key
if "typed_digits" not in st.session_state:
    st.session_state.typed_digits = ""
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

st.write("✏️ Draw a digit (0–9) below:")

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
            st.session_state.typed_digits += digit

            st.success(f"✅ Recognized: {digit}")
            st.write(f"Typed so far: `{st.session_state.typed_digits}`")

            # Clear canvas by changing key and rerunning
            st.session_state.canvas_key += 1
            st.rerun()
        else:
            st.write(f"Typed so far: `{st.session_state.typed_digits}`")
    except Exception as e:
        st.warning(f"Could not process image: {str(e)}")
        st.write(f"Typed so far: `{st.session_state.typed_digits}`")
else:
    st.write(f"Typed so far: `{st.session_state.typed_digits}`")

# Button to reset
if st.button("🔁 Reset typed digits"):
    st.session_state.typed_digits = ""
    st.session_state.canvas_key += 1
    st.rerun()
