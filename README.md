# PneumoAI
# 🩺 PneumoAI – AI-Powered Pneumonia Detection System

PneumoAI is a web-based healthcare application that uses Deep Learning and Explainable AI to detect Pneumonia from Chest X-Ray images.

The system allows users to register, log in, upload Chest X-Ray images, receive AI-powered predictions, and visualize the model's decision using Grad-CAM heatmaps.

---

## 🚀 Features

### User Authentication

* User Registration
* Secure Login & Logout
* Password Hashing using Werkzeug
* Session-based Authentication

### Pneumonia Detection

* Upload Chest X-Ray images
* CNN-based classification
* Predict:

  * NORMAL
  * PNEUMONIA
* Display confidence score

### Explainable AI (XAI)

* Grad-CAM Visualization
* Highlights regions that influenced the CNN prediction
* Improves transparency and interpretability

### Prediction History

* Stores previous predictions
* Displays:

  * Image Name
  * Prediction Result
  * Confidence Score
  * Date & Time

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 🧠 Deep Learning Model

The model is built using TensorFlow/Keras and trained on Chest X-Ray images.

### Architecture

* Conv2D
* MaxPooling2D
* Conv2D
* MaxPooling2D
* Conv2D
* MaxPooling2D
* Flatten
* Dense
* Dropout
* Output Layer (Sigmoid)

### Input Size

128 × 128 × 3

### Output

Binary Classification:

* 0 → NORMAL
* 1 → PNEUMONIA

---

## 📊 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 78.53% |
| Precision | 74.90% |
| Recall    | 98.72% |
| F1 Score  | 85.18% |

### Confusion Matrix

* True Normal: 105
* False Pneumonia: 129
* False Normal: 5
* True Pneumonia: 385

The model achieves very high recall, making it effective for identifying pneumonia cases while minimizing missed diagnoses.

---

## 🛠 Technologies Used

### Backend

* Python
* Flask
* Flask-SQLAlchemy

### Deep Learning

* TensorFlow
* Keras
* NumPy

### Explainable AI

* Grad-CAM
* OpenCV
* Matplotlib

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* JavaScript

### Database

* SQLite

### Version Control

* Git
* GitHub

---

## 📁 Project Structure

```text
PneumoAI/
│
├── app.py
├── train_model.py
├── predict.py
├── evaluate_model.py
├── gradcam.py
├── create_db.py
├── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   └── history.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── uploads/
│   └── confusion_matrix.png
│
├── models/
│   └── pneumonia_model.h5
│
└── dataset/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/K81-create/PneumoAI.git
cd PneumoAI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📸 Application Workflow

1. User opens the landing page.
2. User registers or logs in.
3. User uploads a Chest X-Ray image.
4. CNN model predicts:

   * NORMAL
   * PNEUMONIA
5. Confidence score is displayed.
6. Grad-CAM heatmap is generated.
7. Prediction is stored in the database.
8. User can view prediction history.

---

## 🔮 Future Improvements

* Transfer Learning (DenseNet121 / ResNet50)
* Lung Segmentation
* Multi-class Lung Disease Detection
* Cloud Deployment
* Doctor Dashboard
* PDF Report Generation
* Email Notifications

---

## 👩‍💻 Author

Khushi Singhal

AI & Data Analytics Enthusiast

---

## 📜 License

This project is licensed under the MIT License.
