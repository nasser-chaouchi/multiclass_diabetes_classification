# Multiclass Diabetes Classification Project

![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/ML-sklearn-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

This project aims to classify patients into **three diabetes categories** (Non-Diabetic, Diabetic, Pre-Diabetic) based on clinical and biochemical features. It includes:

- A complete machine learning pipeline built in Python
- A well-documented Jupyter notebook with data exploration, modeling, and evaluation
- A deployed **Streamlit app** for real-time prediction

🔗 **Try the app online**:  
👉 https://multiclassdiabetes-vibkpbkqtd7zdhjbmojppp.streamlit.app/

<img width="386" height="742" alt="image" src="https://github.com/user-attachments/assets/d6a628be-437a-47fe-95d7-8b581841f1dd" />


---

## Project Overview

This project uses a dataset of 264 patients and applies several ML models to classify diabetes status. After evaluation, the **Random Forest** model was selected based on its F1-macro score and generalization ability.

---

## Project Structure

- `dataset/` – Multiclass Diabetes Dataset.csv
- `multiclass_diabetes_classification.ipynb` – Main notebook (EDA + modeling)
- `rf_model.pkl` – Trained Random Forest model (saved with joblib)
- `app.py` – Streamlit app source code
- `requirements.txt` – Python dependencies

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/nasser-chaouchi/multiclass_diabetes_classification.git
cd multiclass-diabetes-classification
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```


## 📊 Model Performance Summary

| Model                | F1-score (macro) |
|----------------------|------------------|
| Logistic Regression  | 0.827            |
| Random Forest        | 0.9718           |
| K-Nearest Neighbors  | 0.6771           |

✅ Final model accuracy on test set: **97%**  
✅ Macro F1-score: **0.98**

---

## Built With

- Python
- scikit-learn
- pandas, seaborn, matplotlib
- Streamlit

---

## 🙋‍♂️ Author

**Nasser Chaouchi**  
💼 Data Scientist  
🌐 [LinkedIn](https://www.linkedin.com/in/nasser-chaouchi/)

## 📄 License
This project is licensed under the MIT License.

