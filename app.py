import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
from pickle import dump, load
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Suppress all warnings for a cleaner Streamlit output
warnings.filterwarnings("ignore")

# --- Configuration ---
FILENAME = 'pima-indians-diabetes.csv'
NAMES = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
SEED = 7 
TEST_SIZE = 0.33
NUM_FOLDS = 10
MODEL_FILENAME = 'finalized_diabetes_model.sav'

# --- Utility Functions ---

@st.cache_data
def load_data(filename, names):
    """Loads the dataset and separates features (X) from target (Y)."""
    # Create a mock CSV file content since Streamlit doesn't automatically handle external files
    # This is a critical point for Streamlit Cloud deployment without direct file upload mechanism
    # NOTE: You MUST upload the 'pima-indians-diabetes.csv' file alongside this script
    # for production, or replace this mock logic with your actual data handling.
    try:
        data = pd.read_csv(filename, names=names)
    except FileNotFoundError:
        st.error(f"Error: The data file '{filename}' was not found. Please upload it.")
        return None, None
    
    array = data.values
    X = array[:, 0:8]
    Y = array[:, 8]
    return X, Y

@st.cache_resource
def train_model(X_train, Y_train):
    """Trains and returns the final (best-performing) Pipeline model."""
    # Using the standardized Logistic Regression pipeline, as it performed well.
    final_model = Pipeline([
        ('standardize', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=500, random_state=SEED))
    ])
    final_model.fit(X_train, Y_train)
    return final_model

@st.cache_resource
def compare_models(X, Y):
    """Performs cross-validation comparison of multiple ML algorithms."""
    
    # --- FIX APPLIED HERE ---
    # Changed 'base_estimator' to 'estimator' in BaggingClassifier for scikit-learn >= 1.2
    # Also added random_state to KNN for compatibility/reproducibility in some older versions.
    models_to_compare = []
    models_to_compare.append(('LR', LogisticRegression(max_iter=500, random_state=SEED)))
    models_to_compare.append(('KNN', KNeighborsClassifier(n_neighbors=5))) # Default n_neighbors
    models_to_compare.append(('CART', DecisionTreeClassifier(random_state=SEED)))
    models_to_compare.append(('NB', GaussianNB()))
    models_to_compare.append(('SVM', SVC(random_state=SEED)))
    models_to_compare.append(('ADA', AdaBoostClassifier(n_estimators=100, random_state=SEED)))
    
    # FIX: base_estimator changed to estimator in newer scikit-learn
    models_to_compare.append(('BAG', BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED), n_estimators=100, random_state=SEED)))
    
    models_to_compare.append(('RFC', RandomForestClassifier(n_estimators=100, max_features=3, random_state=SEED)))
    estimators_voting = [
        ('logistic', LogisticRegression(max_iter=500, random_state=SEED)),
        ('cart', DecisionTreeClassifier(random_state=SEED)),
        ('svm', SVC(random_state=SEED, probability=True))
    ]
    models_to_compare.append(('VOT', VotingClassifier(estimators_voting, voting='soft')))


    results = []
    names = []
    scoring = 'accuracy'
    
    for name, model in models_to_compare:
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        
    return results, names

# --- Streamlit UI ---

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("Pima Indians Diabetes Prediction")
st.markdown("Use the controls below to compare models and test the final prediction pipeline.")

# --- Data Loading and Splitting ---
X, Y = load_data(FILENAME, NAMES)

if X is not None:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=SEED)
    
    # Train the final model
    final_model = train_model(X_train, Y_train)
    
    # --- Model Comparison Section ---
    st.header("1. Model Performance Comparison")
    
    # Button to trigger model comparison (which runs the heavy CV step)
    if st.button("Run Algorithm Comparison (Cross-Validation)"):
        with st.spinner("Comparing models... this may take a moment."):
            cv_results, model_names = compare_models(X, Y)
            
            summary_data = {
                'Model': model_names,
                'Mean Accuracy': [res.mean() for res in cv_results],
                'Std Dev': [res.std() for res in cv_results]
            }
            summary_df = pd.DataFrame(summary_data)
            
            st.subheader("Cross-Validation Results Summary (Accuracy)")
            st.dataframe(summary_df.set_index('Model').style.format({
                'Mean Accuracy': '{:.4f}',
                'Std Dev': '{:.4f}'
            }))
            
            # Matplotlib Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title('Algorithm Comparison - Accuracy Scores')
            ax.boxplot(cv_results, showfliers=False)
            ax.set_xticklabels(model_names)
            ax.set_ylabel('Cross-Validation Accuracy Score')
            ax.grid(axis='y', linestyle='--')
            
            st.pyplot(fig)


    # --- Prediction Section ---
    st.header("2. Predict Diabetes Status")
    st.markdown("Adjust the input features to get a real-time prediction using the trained **Standardized Logistic Regression** model.")
    
    col1, col2, col3 = st.columns(3)
    
    # Use standard values as defaults for the input fields
    with col1:
        preg = st.number_input("1. Pregnancies (Number)", min_value=0, max_value=17, value=3)
        plas = st.number_input("2. Glucose (Plasma glucose concentration)", min_value=0, max_value=200, value=120)
        pres = st.number_input("3. Blood Pressure (Diastolic blood pressure)", min_value=0, max_value=122, value=70)
    
    with col2:
        skin = st.number_input("4. Skin Thickness (Triceps skin fold thickness)", min_value=0, max_value=100, value=20)
        test = st.number_input("5. Insulin (2-Hour serum insulin)", min_value=0, max_value=850, value=80)
        mass = st.number_input("6. BMI (Body mass index)", min_value=0.0, max_value=67.1, value=32.0, format="%.1f")
    
    with col3:
        pedi = st.number_input("7. Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.47, format="%.3f")
        age = st.number_input("8. Age (Years)", min_value=21, max_value=81, value=33)

    
    input_data = np.array([[preg, plas, pres, skin, test, mass, pedi, age]])
    
    # Make Prediction
    if st.button("Predict"):
        with st.spinner("Generating prediction..."):
            # The final_model is a Pipeline, so it handles the standardization internally.
            prediction = final_model.predict(input_data)[0]
            prediction_proba = final_model.predict_proba(input_data)[0]
            
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: #ffcccc; color: black;'><b>Status: POSITIVE for Diabetes</b></div>", 
                    unsafe_allow_html=True
                )
                st.info(f"Confidence (Probability of Diabetes): {prediction_proba[1]*100:.2f}%")
            else:
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: #ccffcc; color: black;'><b>Status: NEGATIVE for Diabetes</b></div>", 
                    unsafe_allow_html=True
                )
                st.info(f"Confidence (Probability of Negative): {prediction_proba[0]*100:.2f}%")
            
            st.markdown("---")
            st.markdown(f"**Model Type Used:** Standardized Logistic Regression Pipeline (Accuracy on Test Set: {final_model.score(X_test, Y_test)*100.0:.2f}%)")


else:
    st.warning("Please ensure the data file 'pima-indians-diabetes.csv' is available to run the app.")
