# One-to-One RNN
# Handwritten Digit Recognition using SimpleRNN

import os
import numpy as np
import streamlit as st

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Configuration

MODEL = "digit_rnn.keras"

# Train Model

def train_model():

    print("Loading MNIST Dataset...")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Training Images:", x_train.shape)
    print("Testing Images :", x_test.shape)

    # Normalize

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Convert labels to one-hot encoding

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build SimpleRNN Model

    model = Sequential()

    model.add(
        SimpleRNN(
            128,
            input_shape=(28, 28),
            activation="tanh"
        )
    )

    model.add(
        Dense(
            64,
            activation="relu"
        )
    )

    model.add(
        Dense(
            10,
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2
    )
    # Save Model

    model.save(MODEL)

    # Evaluate

    loss, accuracy = model.evaluate(x_test, y_test)

    print("\nTest Accuracy:", accuracy)

import cv2

def predict_digit(image):

    @st.cache_resource
    def load_saved_model():
        return load_model(MODEL)

    model = load_saved_model()

    # Convert PIL image to grayscale
    image = image.convert("L")

    img = np.array(image)

    # Resize while keeping quality
    img = cv2.resize(img, (280, 280))

    # Blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Threshold
    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, 0

    # Largest contour
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    digit = img[y:y+h, x:x+w]

    # Create square canvas
    size = max(w, h) + 20

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size-h)//2
    x_offset = (size-w)//2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = digit

    # Resize to MNIST size
    square = cv2.resize(square, (28,28))

    # Normalize
    square = square.astype("float32") / 255.0

    # Show processed image
    st.image(
        square,
        caption="Processed Image",
        width=180
    )

    # Prediction
    prediction = model.predict(
        square.reshape(1,28,28),
        verbose=0
    )[0]

    predicted_digit = np.argmax(prediction)

    confidence = prediction[predicted_digit]

    return predicted_digit, confidence

# Train only once

if not os.path.exists(MODEL):
    train_model()


# ======================================================
# STREAMLIT UI 
# ======================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background:linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

[data-testid="stHeader"]{
background:transparent;
}

.block-container{
padding-top:2rem;
max-width:1100px;
}

h1,h2,h3,h4,h5,h6,p,label{
color:white !important;
}

.title{
text-align:center;
font-size:60px;
font-weight:800;
color:white;
margin-bottom:8px;
}

.subtitle{
text-align:center;
font-size:22px;
color:#cbd5e1;
margin-bottom:25px;
}


.upload-box{
background:#1f2937;
padding:25px;
border-radius:18px;
border:1px solid #334155;
box-shadow:0px 8px 20px rgba(0,0,0,.35);
}

.result-card{
background:linear-gradient(135deg,#2563eb,#4f46e5);
padding:30px;
border-radius:18px;
text-align:center;
margin-top:25px;
color:white;
box-shadow:0px 8px 25px rgba(0,0,0,.4);
}

.footer{
text-align:center;
color:#94a3b8;
margin-top:40px;
font-size:15px;
}

.stButton>button{
width:100%;
height:58px;
font-size:20px;
font-weight:bold;
border:none;
border-radius:14px;
color:white;
background:linear-gradient(90deg,#2563eb,#4f46e5);
}

.stButton>button:hover{
background:linear-gradient(90deg,#1d4ed8,#4338ca);
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("📊 Model Information")

    st.markdown("""
**🧠 Model**
- SimpleRNN

**🎯 Task**
- Handwritten Digit Recognition

**📂 Dataset**
- MNIST

**📤 Input**
- Handwritten Digit Image

**📥 Output**
- Predicted Digit

**⚙ Framework**
- TensorFlow + Keras

**💻 Frontend**
- Streamlit
""")

    st.divider()

    st.success(
        "💡 Upload a handwritten digit image and click **Predict Digit**."
    )

# ---------------- HEADER ----------------

st.markdown(
"""
<div class='title'>
🧠 Handwritten Digit Recognition
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
One-to-One RNN using <b>SimpleRNN</b>
</div>
""",
unsafe_allow_html=True
)

st.info(
    "📤 Upload a handwritten digit image (0–9). The trained SimpleRNN model will recognize the digit and display its confidence score."
)

uploaded_file = st.file_uploader(
    "📂 Upload a Handwritten Digit Image",
    type=["png", "jpg", "jpeg"],
    label_visibility="visible"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        st.write(f"**Image Size:** {image.size[0]} × {image.size[1]}")
        st.write(f"**Mode:** {image.mode}")

    with col2:

        st.markdown("### ⚙ Image Information")

        gray = image.convert("L")

        st.image(
            gray,
            caption="Grayscale Preview",
            use_container_width=True
        )

        st.write("✔ Converted to grayscale")
        st.write("✔ Ready for preprocessing")
        st.write("✔ MNIST format: 28 × 28")

    st.write("")

    predict = st.button("🔍 Predict Digit")

else:

    predict = False

if predict:

    with st.spinner("🧠 Recognizing digit..."):

        digit, confidence = predict_digit(image)

    if digit is None:

        st.error("❌ No handwritten digit detected. Please upload a clearer image.")

    else:

        st.success("✅ Prediction Completed Successfully!")

        st.markdown(
            f"""
            <div class='result-card'>
                <h2 style="margin:0;font-size:22px;">
                    🎯 DIGIT RECOGNIZED
                </h2>

                <h1 style="font-size:90px;margin:10px 0;color:white;">
                    {digit}
                </h1>

                <h3 style="margin:0;">
                    Confidence : {confidence*100:.2f}%
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.subheader("📊 Prediction Confidence")

        st.progress(float(confidence))

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Predicted Digit",
                digit
            )

        with c2:

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        if confidence >= 0.95:

            st.success("🌟 High Confidence Prediction")

        elif confidence >= 0.80:

            st.info("👍 Good Prediction Confidence")

        else:

            st.warning("⚠ Low Confidence. Try uploading a clearer image.")

# ======================================================
# SAMPLE DIGITS
# ======================================================

with st.expander("📌 Tips for Better Predictions"):

    st.markdown("""
- ✍ Write one digit only.
- ⚪ Use a white background.
- ⚫ Write the digit in dark ink.
- 📷 Crop the image close to the digit.
- 📐 Keep the digit centered.
- 🚫 Avoid shadows and blur.
""")

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown(
"""
<div class='footer'>
❤️ Built with TensorFlow • Keras • SimpleRNN • Streamlit
</div>
""",
unsafe_allow_html=True
)