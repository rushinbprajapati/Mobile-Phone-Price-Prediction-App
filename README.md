Mobile Phone Price Prediction (Streamlit App)
markdown
Copy
Edit
# 📱 Mobile Phone Price Prediction App

This is a Streamlit web app that predicts the **price range** of a mobile phone based on its specifications using a Logistic Regression model.

---

## 🚀 Features

- User-friendly sliders and checkboxes to input mobile features
- Logistic Regression model trained on `dataset.csv`
- Predicts one of four categories: Low Cost, Medium Cost, High Cost, Very High Cost
- Displays class probabilities and prediction results

---

## 🧰 Requirements

- Python 3.8 or above
- Streamlit
- scikit-learn
- pandas
- numpy

---

## 🏗️ Project Setup

### Step 1: Clone or Download the Project

```bash
git clone https://github.com/yourusername/mobile-price-predictor.git
cd mobile-price-predictor
Or manually download and extract the .zip file into a folder.

Step 2: Create a Virtual Environment
Windows:

bash
Copy
Edit
python -m venv .venv
.venv\Scripts\activate
Mac/Linux:

bash
Copy
Edit
python3 -m venv .venv
source .venv/bin/activate
Step 3: Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt doesn’t exist, you can install manually:

bash
Copy
Edit
pip install streamlit pandas scikit-learn numpy
Step 4: Ensure Dataset is Present
Place your dataset.csv file in the same folder as app.py.

Step 5: Run the App
bash
Copy
Edit
streamlit run app.py"# Mobile-Phone-Price-Prediction-App" 
