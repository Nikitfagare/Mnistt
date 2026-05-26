import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load trained CNN model
model = tf.keras.models.load_model("mnist_cnn.h5")

# App title
st.title("CNN Handwritten Digit Recognition")

st.write("Upload an image of a handwritten digit (0-9)")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert('L')

    # Display uploaded image
    st.image(image, caption="Uploaded Image", width=200)

    # Resize image to 28x28
    image = image.resize((28, 28))

    # Convert image to numpy array
    img_array = np.array(image)

    # Invert colors (optional for white background images)
    img_array = 255 - img_array

    # Normalize pixel values
    img_array = img_array / 255.0

    # Reshape for CNN input
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array)

    # Get predicted digit
    predicted_digit = np.argmax(prediction)

    # Confidence score
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f"Predicted Digit: {predicted_digit}")

    st.write(f"Confidence: {confidence:.2f}%")

    # Show probability for all digits
    st.subheader("Prediction Probabilities")

    for i in range(10):
        st.write(f"Digit {i}: {prediction[0][i]*100:.2f}%")
