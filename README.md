
# 🧠 AI Medical Chatbot for Infectious & Disease Prediction

An AI-powered medical chatbot that predicts infectious diseases and provides personalized health recommendations using machine learning and natural language processing. This chatbot supports early detection, public health awareness, and efficient decision-making during health crises.

---

## 📌 Project Overview

This project demonstrates how AI and deep learning can enhance user interaction in the medical field. The chatbot utilizes a deep feedforward multilayer perceptron model trained on medical text and image datasets to assess disease symptoms and provide guidance.

### 🎯 Key Features

- Predicts infectious diseases from symptoms and MRI images
- Uses NLP to simulate human-like dialogue
- Provides tailored health advice and educational responses
- Flask-based web interface with support for voice/text queries
- Integration of disease-specific datasets (e.g., diabetes, cancer, malaria)

---

## 🏗️ System Architecture

1. **Frontend**: HTML/CSS interface powered by Flask
2. **Backend**:
   - Python with Flask
   - Disease prediction using Convolutional Neural Networks (CNN)
   - Symptom classification via TF-IDF + cosine similarity
3. **Datasets**:
   - MRI images for dementia detection
   - 15+ text corpora for common diseases

---

## 🧠 AI Techniques Used

- **Multilayer Perceptron (MLP)** for prediction logic
- **CNN** for dementia image classification
- **TF-IDF Vectorization** for symptom processing
- **Cosine Similarity** for chatbot response retrieval
- Custom rule-based logic for intent classification

---

## 🔧 Requirements

### Hardware
- Laptop with Intel processor
- Minimum 512MB RAM
- 40GB storage

### Software
- OS: Windows 11
- Language: Python 3.x
- Libraries: Flask, NLTK, TensorFlow/Keras, Sklearn
- Frontend: HTML, CSS

---

## 🚀 Getting Started

1. **Clone this repository**:
   ```bash
   git clone https://github.com/haashu0412/Medical_Chatbot.git
   cd ai-medical-chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**:
   ```bash
   python app.py
   ```

4. **Access the chatbot**:
   Open `http://127.0.0.1:5000` in your browser.

---

## 📊 Sample Output

- Text-based queries:
  - `User:` "I have a fever and sore throat"
  - `Bot:` "This could be a serious symptom.. Please consult a Doctor ASAP."
    ![image](https://github.com/user-attachments/assets/15b7e4ff-355c-405b-bcd5-9585f852da17)


- Image-based prediction:
  - Upload an MRI image
  - Bot returns disease stage classification (e.g., "Mild Demented", "Normal")
    ![image](https://github.com/user-attachments/assets/327c40e4-4c01-434f-8c17-6fdff4905d68)


---

## 📈 Evaluation Metrics

- **Accuracy:** Up to 94.32%
- **Loss:** Minimum observed loss: 0.1232
- **Confusion Matrix, F1 Score, Precision/Recall** used for classification evaluation
- **User Satisfaction:** Improved trust and efficiency observed in mock tests

---

## 📌 Future Work

- Integration with Electronic Health Records (EHR)
- Multilingual voice-based interaction
- Expanded dataset support
- Enhanced accuracy and ethics compliance

---


