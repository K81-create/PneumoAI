import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("models/pneumonia_model.h5")

# Load image
img = image.load_img("sample.jpeg", target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("PNEUMONIA")
    print("Confidence:", prediction[0][0])
else:
    print("NORMAL")
    print("Confidence:", 1 - prediction[0][0])