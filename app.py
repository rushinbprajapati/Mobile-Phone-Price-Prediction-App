import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# --- 1. Load Data and Train Model (Cached) ---
@st.cache_resource
def load_data_and_train_model():
    """
    Loads the dataset, trains the Logistic Regression model, and returns
    the trained model, feature names, and target names.
    This function is cached to run only once.
    """
 
    try:
        df = pd.read_csv('dataset.csv')
    except FileNotFoundError:
        st.error("Error: 'dataset.csv' not found. Please ensure the file is in the same directory as this Streamlit app.")
        st.stop()

    # Define Features (X) and Target (y)
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    # Price range mapping
    price_range_names = {
        0: 'Low Cost',
        1: 'Medium Cost',
        2: 'High Cost',
        3: 'Very High Cost'
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train, y_train)

    return model, X.columns.tolist(), price_range_names

# Load model and metadata
model, feature_names, price_range_map = load_data_and_train_model()
target_names = list(price_range_map.values())

# --- 2. Page Setup ---
st.set_page_config(page_title="Mobile Phone Price Predictor", layout="centered")
st.title("📱 Mobile Phone Price Prediction App")
st.markdown("""
This app predicts the price range (Low, Medium, High, Very High) of a mobile phone
based on its specifications. The model is trained directly within the app.
""")
st.write("---")

# --- 3. User Input Section ---
st.header("Enter Mobile Phone Specifications")
input_features = {}
col1, col2 = st.columns(2)

with col1:
    input_features['battery_power'] = st.slider("Battery Power (mAh)", 500, 6000, 3000, 50)
    input_features['clock_speed'] = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5, 0.1)
    input_features['int_memory'] = st.slider("Internal Memory (GB)", 0, 128, 32, 1)
    input_features['ram'] = st.slider("RAM (MB)", 256, 12000, 4000, 256)
    input_features['pc'] = st.slider("Primary Camera (MP)", 0, 20, 8, 1)
    input_features['fc'] = st.slider("Front Camera (MP)", 0, 20, 5, 1)
    input_features['px_height'] = st.slider("Pixel Resolution Height", 0, 2000, 1000, 10)
    input_features['px_width'] = st.slider("Pixel Resolution Width", 0, 2000, 1000, 10)
    input_features['talk_time'] = st.slider("Talk Time (hours)", 0, 20, 10, 1)

with col2:
    input_features['mobile_wt'] = st.slider("Mobile Weight (gm)", 80, 200, 150, 1)
    input_features['n_cores'] = st.slider("Processor Core Count", 1, 8, 4, 1)
    input_features['sc_h'] = st.slider("Screen Height (cm)", 5.0, 20.0, 12.0, 0.1)  # FIXED
    input_features['sc_w'] = st.slider("Screen Width (cm)", 0.0, 20.0, 5.0, 0.1)   # FIXED
    input_features['m_dep'] = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5, 0.01)

    st.write("---")
    st.subheader("Connectivity & Features")
    input_features['blue'] = st.checkbox("Bluetooth", value=True)
    input_features['dual_sim'] = st.checkbox("Dual SIM", value=True)
    input_features['four_g'] = st.checkbox("4G Support", value=True)
    input_features['three_g'] = st.checkbox("3G Support", value=True)
    input_features['touch_screen'] = st.checkbox("Touch Screen", value=True)
    input_features['wifi'] = st.checkbox("Wi-Fi", value=True)

# Convert boolean values to integers
for key in ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']:
    input_features[key] = 1 if input_features[key] else 0

# Prepare input data in correct format
input_df = pd.DataFrame([input_features])
input_df = input_df[feature_names]  # Keep same feature order as training

st.write("---")

# --- 4. Prediction ---
if st.button("Predict Mobile Price Range"):
    prediction_proba = model.predict_proba(input_df)
    prediction = model.predict(input_df)

    predicted_price_index = prediction[0]
    predicted_price_name = price_range_map.get(predicted_price_index, "Unknown")

    st.header("Prediction Result")
    st.success(f"The predicted mobile phone price range is: **{predicted_price_name}**")

    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame(prediction_proba, columns=target_names)
    st.dataframe(proba_df.style.format("{:.2%}"))
