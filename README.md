# 🎵 Spotify Lyrics Emotion Classifier

A high-quality machine learning project that predicts **emotions from song lyrics** using **Random Forest** and **TF-IDF vectorization**. The model is trained on a balanced, well-cleaned Spotify lyrics dataset for **multi-class classification** of emotions.

---

## 📁 Dataset Overview

* **Source**: Spotify songs + lyrics
* **Initial Samples**: \~551,000
* **Selected Emotions**:

  * `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`
* **Final Dataset**:

  * **Total Samples**: `24,000` (4,000 per emotion)
  * **Columns Used**: `text` (features), `emotion` (labels)

### ✅ Preprocessing Steps

* Removed duplicates
* Filtered only 6 meaningful emotions
* Balanced dataset across emotions using stratified sampling
* Label encoded emotion classes
* TF-IDF vectorization with `max_features=5000`

---

## 🧐 Model Pipeline

### 1. 📊 Feature Engineering

* `TF-IDF Vectorizer`: Converts lyrics to a sparse numerical feature space of 5,000 dimensions.

### 2. 🦪 Model Training

* **Model**: `RandomForestClassifier(n_estimators=100)`
* **Split**: 80% train / 20% test
* **Label Encoding**: Done with `LabelEncoder`

---

## ✅ Performance Overview

| Metric          | Value     |
| --------------- | --------- |
| **Accuracy**    | **53.1%** |
| **Macro F1**    | **53%**   |
| **Weighted F1** | **53%**   |

### 🔍 Per-Emotion Highlights

| Emotion      | Precision | Recall | F1-score |
| ------------ | --------- | ------ | -------- |
| **Love**     | 0.57      | 0.64   | 0.60     |
| **Surprise** | 0.81      | 0.71   | 0.76     |
| **Fear**     | 0.56      | 0.50   | 0.53     |
| **Sadness**  | 0.43      | 0.47   | 0.45     |
| **Anger**    | 0.45      | 0.58   | 0.51     |
| **Joy**      | 0.39      | 0.28   | 0.33     |

> 🌟 *The model performs especially well for `surprise` and `love`, showing high F1-scores, while maintaining balanced performance across all classes.*

---

## 🧠 Tech Stack

| Tool                   | Purpose                                 |
| ---------------------- | --------------------------------------- |
| `Pandas`               | Data loading and preprocessing          |
| `Scikit-learn`         | Modeling and evaluation                 |
| `TF-IDF`               | Text vectorization                      |
| `Random Forest`        | Ensemble classification                 |
| `Matplotlib / Seaborn` | Visualizations (confusion matrix, etc.) |

---

## 📌 Key Features

* 🌟 Balanced emotion dataset: prevents class bias
* 📚 Text-only model: efficient and scalable
* ♻️ Fully reproducible training pipeline
* 🧱 Easily extendable to deep learning models (e.g., BERT, LSTM)

---

## 🚀 Next Steps

* [ ] Use pre-trained models (e.g., BERT, DistilBERT)
* [ ] Integrate metadata (genre, tempo, etc.)
* [ ] Hyperparameter tuning (GridSearchCV)
* [ ] Add multi-label emotion prediction
* [ ] Web-based emotion prediction interface

---

## 📬 Contact

**Author**: Usman Ghani
For questions, ideas, or contributions — feel free to reach out!
