import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")
st.title("✏️ Draw & Convert: Handwritten Digit Visualizer")

st.markdown("""
Draw a digit in the canvas below.  
This app converts your drawing into a **28×28 grayscale image** (like MNIST) and displays the pixel values.
""")

# --- DRAWING CANVAS ---
st.subheader("✏️ Draw a digit by hand:")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
)

# --- IF SOMETHING IS DRAWN ---
if canvas_result.image_data is not None:
    # Convert canvas to PIL image
    drawn_img = Image.fromarray((canvas_result.image_data).astype("uint8"))
    gray_img = drawn_img.convert("L")  # convert to grayscale

    st.subheader("🖼️ Handwritten Digit:")
    st.image(gray_img, width=300)

    # --- Resize to 28x28 ---
    img_gray_resized = gray_img.resize((28, 28))
    pixel_array = np.array(img_gray_resized)

    # --- Color inversion toggle (for ML models like MNIST) ---
    invert = st.checkbox("Invert colors (black background for digit)")
    if invert:
        pixel_array = 255 - pixel_array
        img_gray_resized = Image.fromarray(pixel_array)

    # --- Normalize toggle ---
    normalize = st.checkbox("Normalize pixel values (0–1)")
    if normalize:
        pixel_array = pixel_array / 255.0

    # --- Show resized digit ---
    st.subheader("🔲 Resized Digit (28×28):")
    st.image(img_gray_resized, width=200)

    st.markdown("The pixel values range from **0 (black)** to **255 (white)** unless normalized.")

    # --- Show pixel grid ---
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_gray_resized, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks(np.arange(0, 28, 1))
    ax.set_yticks(np.arange(0, 28, 1))
    ax.grid(color="red", linewidth=0.3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    st.pyplot(fig)

    # --- Display numeric pixel array ---
    st.subheader("🧮 Pixel Values (28×28):")
    st.dataframe(pixel_array)

    # --- Download button for processed image ---
    buf = BytesIO()
    img_gray_resized.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        "💾 Download 28×28 Image",
        data=byte_im,
        file_name="digit_28x28.png",
        mime="image/png"
    )
else:

    st.info("Draw a digit in the canvas above to begin.")
