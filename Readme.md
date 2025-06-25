# Malaria Diagnosis Using CNN
**Model**<img src= "https://github.com/anasdev-10/malaria-diagnosis/blob/main/y3042gtk.png"
style="width:6.85039in;height:4.56693in" />



This project implements a Convolutional Neural Network (CNN) to automatically classify malaria cell images as either parasitized or uninfected. The model is built and trained using TensorFlow and Google Colab to support rapid diagnosis and improve healthcare outcomes.

---

## 🚀 Project Overview
- Developed a CNN model for binary image classification.
- Utilized the malaria dataset from TensorFlow Datasets (TFDS).
- Achieved effective model training with image preprocessing and performance evaluation.

---

## 📂 Dataset
- **Source:** TensorFlow Datasets (`tfds.load('malaria')`)
- **Classes:** Parasitized and Uninfected
- **Input Format:** RGB images with corresponding binary labels

---

##  Tech Stack
- Python
- TensorFlow
- Keras
- TensorFlow Datasets (TFDS)
- Matplotlib, Seaborn for visualization
- Google Colab

---

## 📊 Key Features
- Image preprocessing and batch loading.
- CNN model architecture with dropout and batch normalization.
- Performance monitoring using accuracy, precision, recall, and confusion matrix.
- Visualization of ROC curves and training metrics.

---

## ✅ Results
- The model successfully distinguishes between parasitized and uninfected cells with good classification performance.
- Plots of training accuracy and loss provided in the notebook.

---

## 💻 How to Run
1. Open the Google Colab notebook.
2. Ensure `tensorflow_datasets` is installed (Google Colab typically has it pre-installed).
3. Run all cells sequentially to train the model and visualize results.

---

## 🔍 Future Improvements
- Hyperparameter tuning for further accuracy improvement.
- Experimenting with advanced architectures like ResNet.
- Deploying as a web app using Streamlit for broader accessibility.

---

> This project applies deep learning techniques to contribute to healthcare solutions through image-based malaria diagnosis.
