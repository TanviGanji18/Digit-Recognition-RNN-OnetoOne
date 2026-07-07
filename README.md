# One-to-One RNN Handwritten Digit Recognition

## Overview

This project demonstrates a **One-to-One Recurrent Neural Network (SimpleRNN)** for handwritten digit recognition using the **MNIST** dataset. The model is trained to classify handwritten digits (0–9) and provides predictions through an interactive **Streamlit** web application.

## Features

- One-to-One RNN architecture
- Handwritten digit recognition
- Built using TensorFlow/Keras
- Uses the MNIST dataset
- Interactive Streamlit interface
- Automatic model training and saving
- Displays predicted digit with confidence score

## Dataset

**Dataset:** MNIST Handwritten Digits

The MNIST dataset contains 70,000 grayscale images of handwritten digits.

- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Classes: 10 (Digits 0–9)

The dataset is automatically downloaded from TensorFlow during the first execution.

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Project Structure

```
OneToOne-RNN-Digit-Recognition/
│
├── app.py
├── digit_rnn.keras
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project folder:

```bash
cd OneToOne-RNN-Digit-Recognition
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## How It Works

1. Load the MNIST dataset.
2. Normalize the image pixel values.
3. Train a SimpleRNN model on handwritten digits.
4. Save the trained model.
5. Upload a handwritten digit image.
6. Predict the digit using the trained model.
7. Display the predicted digit and confidence score.

## Example

### Input

Upload a handwritten digit image.

### Output

```
Predicted Digit: 7

Confidence: 99.43%
```

## Future Improvements

- Add a drawing canvas to draw digits directly.
- Replace SimpleRNN with LSTM or GRU.
- Improve image preprocessing for higher accuracy.
- Deploy the application using Streamlit Community Cloud.

## Author

**One-to-One RNN Handwritten Digit Recognition**  
Developed as a Deep Learning practical project using **SimpleRNN**, **TensorFlow**, and **Streamlit**.