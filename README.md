# 🔤 Letter Recognition: Ensemble Learning vs Deep Learning

This project compares the performance and consistency of **Classical Machine Learning, Ensemble Learning, and Deep Learning (MLP)** models for multi-class letter recognition. The experiments use two different data representations: **feature-based data** from UCI and **raw pixel-based image data** from Hugging Face.

The main goal is to evaluate how different models perform when the same classification task is represented using **statistical features versus raw image pixels**.

---

## 📊 Dataset

Two versions of the Letter Recognition dataset were used:

* **Feature-Based Dataset:** [Letter Recognition – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/59/letter+recognition)

  * 20,000 instances
  * 16 numerical features
  * 26 letter classes (A–Z)

* **Pixel-Based Dataset:** [Letter Recognition – Hugging Face](https://huggingface.co/datasets/pittawat/letter_recognition)

  * Grayscale images
  * 28 × 28 pixels
  * 784 pixel features after flattening

---

## 🔧 Models & Methodology

The project evaluates several classification algorithms across both data representations:

* **Classical ML:** SVM, KNN, Naive Bayes
* **Ensemble Learning:** Random Forest, XGBoost, LightGBM
* **Deep Learning:** Multi-Layer Perceptron (MLP)

### Preprocessing & Evaluation

* Duplicate handling and feature-label separation
* Pixel flattening and normalization
* **80:20 stratified train-test split**
* **5-Fold Stratified Cross-Validation**
* **GridSearchCV** for hyperparameter tuning
* Evaluation using **Accuracy, Macro F1-Score, ROC-AUC, and computational performance**

---

## 📈 Key Results

### Feature-Based Dataset (UCI)

Ensemble models performed strongly on the 16 statistical features:

| Model         | Accuracy |
| ------------- | -------: |
| LightGBM      |   95.79% |
| Random Forest |   95.54% |
| XGBoost       |   95.35% |
| MLP           |   94.29% |
| SVM           |   93.69% |
| KNN           |   93.54% |

### Pixel-Based Dataset (Hugging Face)

Performance changed when models were applied to 28×28 raw pixel data:

| Model    | Accuracy |
| -------- | -------: |
| SVM      |   93.11% |
| LightGBM |   92.72% |
| MLP      |   91.41% |

---

## 💡 Analysis

The results show that model performance depends strongly on **data representation**. Ensemble methods performed particularly well on structured statistical features, while SVM maintained relatively stable performance when moving from feature-based to high-dimensional pixel data.

The comparison also demonstrates that **deep learning is not automatically superior for every image-based classification task**, especially when using a simple MLP without convolutional layers.

Overall, the project highlights the importance of considering both **model architecture and data representation** when selecting an approach for classification problems.

---

## 🛠️ Technologies

* Python
* Scikit-learn
* XGBoost
* LightGBM
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Google Colab
* T4 GPU

https://drive.google.com/drive/folders/1EpIPQAwMQlNNBaPqQAd6BcQ7WOUxfath?ths=true
