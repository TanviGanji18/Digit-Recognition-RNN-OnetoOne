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

# Predict Digit

def predict_digit(image):

    @st.cache_resource
    def load_saved_model():
        return load_model(MODEL)
    model = load_saved_model()

    # Convert image to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    image = np.array(image)

    # Invert colors (MNIST has white digit on black background)
    image = 255 - image

    # Normalize
    image = image.astype("float32") / 255.0

    # Reshape for RNN
    image = image.reshape(1, 28, 28)

    # Prediction
    prediction = model.predict(image, verbose=0)[0]

    digit = np.argmax(prediction)

    confidence = prediction[digit]

    return digit, confidence


# Train only once

if not os.path.exists(MODEL):
    train_model()


# ======================================================
# Streamlit UI
# ======================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🧠",
    layout="centered"
)

# ---------- Custom CSS ----------

st.markdown("""
<style>

.main{
    background: linear-gradient(to right,#eef2ff,#f8fbff);
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#1f4e79;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:25px;
}

.card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 4px 18px rgba(0,0,0,0.15);
}

.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#2563eb,#4f46e5);
    color:white;
    border:none;
    border-radius:12px;
    height:3.2em;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#1d4ed8,#4338ca);
    color:white;
}

.result{
    background:#f1f8ff;
    padding:15px;
    border-radius:12px;
    border-left:6px solid #2563eb;
    margin-top:15px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:25px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------- Header ----------

st.markdown(
    "<div class='title'>🧠 Handwritten Digit Recognition</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>One-to-One RNN using <b>SimpleRNN</b></div>",
    unsafe_allow_html=True
)

st.info(
    "📤 Upload a handwritten digit image (0–9). The trained SimpleRNN model will predict the digit."
)

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1,col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.write("### Image Details")

        st.write("**Mode:**", image.mode)

        st.write("**Size:**", image.size)

        st.write("**Format:**", image.format)

    st.write("")

    if st.button("🔍 Predict Digit"):

        with st.spinner("Predicting..."):

            digit, confidence = predict_digit(image)

        st.success("Prediction Completed Successfully!")

        st.markdown(
            f"""
            <div class='result'>
            <h2 style='text-align:center;color:#1f4e79;'>
            Predicted Digit : {digit}
            </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### Confidence Score")

        st.progress(float(confidence))

        st.metric(
            "Model Confidence",
            f"{confidence*100:.2f}%"
        )

st.markdown("---")

st.markdown(
"""
<div class='footer'>
❤️ Built with TensorFlow • Keras • SimpleRNN • Streamlit
</div>
""",
unsafe_allow_html=True
)