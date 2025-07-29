# Heart Failure Prediction Project

## Project Overview
This project develops a machine learning model to predict the likelihood of heart failure (DEATH_EVENT) in patients based on various clinical and demographic features. The solution includes a Python-based machine learning pipeline for data preprocessing, feature engineering, model training (using XGBoost), evaluation, and a simple Flask web application for real-time predictions. The goal is to provide a tool that can assist medical professionals in identifying high-risk patients, potentially enabling earlier interventions and improved patient outcomes.

## Problem Statement
Heart failure is a critical and widespread cardiovascular condition that significantly impacts global health. Early and accurate identification of individuals at high risk of heart failure is crucial for timely medical intervention, personalized treatment plans, and improved patient prognosis. However, traditional diagnostic methods can be time-consuming and may not always capture the complex interplay of various physiological factors that contribute to heart failure. There is a need for an automated, data-driven approach to assist in risk assessment.

## Project Objective
The primary objectives of this project are:

- **Develop a Robust Prediction Model:** Train and evaluate a machine learning model (XGBoost) capable of accurately predicting heart failure events based on clinical data.
- **Implement Feature Engineering:** Create new, more informative features from the raw dataset to enhance the model's predictive power.
- **Establish a Scalable Prediction Service:** Build a user-friendly web application using Flask that allows for real-time predictions, making the model accessible and actionable for potential use cases.
- **Achieve High Predictive Performance:** Attain a high accuracy and F1-score in predicting heart failure events, ensuring the model's reliability.
- **Provide Clear Interpretability:** Document the features used and their meaning to ensure the model's predictions can be understood in a clinical context.

## Column Dictionary
This section provides a detailed description of each column in the dataset, including both the original features and the engineered features.

### Original Dataset Columns

| Column Name | Description | Type |
|-------------|-------------|------|
| age | Age of the patient (years). | Numerical |
| anaemia | Boolean indicating if the patient has anaemia (0 = No, 1 = Yes). | Binary |
| creatinine_phosphokinase | Level of the CPK enzyme in the blood (mcg/L). Elevated levels can indicate muscle or heart damage. | Numerical |
| diabetes | Boolean indicating if the patient has diabetes (0 = No, 1 = Yes). | Binary |
| ejection_fraction | Percentage of blood leaving the heart with each contraction (%). Lower values indicate weaker heart function. | Numerical |
| high_blood_pressure | Boolean indicating if the patient has high blood pressure (0 = No, 1 = Yes). | Binary |
| platelets | Platelet count in the blood (kiloplatelets/mL). | Numerical |
| serum_creatinine | Level of serum creatinine in the blood (mg/dL). Higher values indicate impaired kidney function. | Numerical |
| serum_sodium | Level of serum sodium in the blood (mEq/L). Low levels (hyponatremia) can be a sign of heart failure. | Numerical |
| sex | Gender of the patient (0 = Female, 1 = Male). | Binary |
| smoking | Boolean indicating if the patient smokes (0 = No, 1 = Yes). | Binary |
| time | Follow-up period (days). This column was dropped as it represents the observation time, not a predictive feature for the event itself. | Numerical |
| DEATH_EVENT | Target Variable: Boolean indicating if the patient died during the follow-up period (0 = No, 1 = Yes). | Binary |

### Engineered Features
These features were created to capture potential interactions and ratios between existing variables, which might provide more predictive power to the model.

| Column Name | Description | Type | Calculation |
|-------------|-------------|------|-------------|
| creatinine_sodium_ratio | Ratio of serum creatinine to serum sodium, potentially indicating kidney and fluid balance issues. | Numerical | serum_creatinine / serum_sodium |
| cpk_platelet_ratio | Ratio of creatinine phosphokinase to platelets, exploring the relationship between muscle damage and clotting ability. | Numerical | creatinine_phosphokinase / platelets |
| anaemia_diabetes | Interaction term between anaemia and diabetes, indicating the presence of both conditions. | Binary | anaemia * diabetes |
| bp_smoking | Interaction term between high blood pressure and smoking, highlighting the combined risk factors. | Binary | high_blood_pressure * smoking |
| age_ejection_interaction | Interaction term between age and ejection fraction, exploring how age might influence heart pump efficiency. | Numerical | age * ejection_fraction |

## Model Training
The predictive model is built using XGBoost (Extreme Gradient Boosting), a highly efficient and powerful machine learning algorithm known for its speed and performance.

### Training Steps:
1. **Data Loading:** The heart_failure_clinical_records_dataset.csv dataset is loaded using Pandas.

2. **Feature Engineering:** Five new features are engineered to capture complex relationships within the data:
   - creatinine_sodium_ratio
   - cpk_platelet_ratio
   - anaemia_diabetes
   - bp_smoking
   - age_ejection_interaction

3. **Feature Selection:** The DEATH_EVENT column is designated as the target variable (y), and the time column is dropped from the features (X) as it represents the follow-up duration rather than a predictive input.

4. **Train-Test Split:** The dataset is split into training (80%) and testing (20%) sets using sklearn.model_selection.train_test_split. A random_state is set for reproducibility, and stratify=y ensures that the proportion of DEATH_EVENT (heart failure) cases is maintained in both training and testing sets.

5. **Feature Scaling:** StandardScaler from sklearn.preprocessing is used to standardize the numerical features. This is crucial for tree-based models like XGBoost, as it helps in faster convergence and prevents features with larger scales from dominating the learning process. The scaler is fit_transform on the training data and only transform on the test data to avoid data leakage.

6. **Model Initialization and Training:** An XGBClassifier is initialized with the following hyperparameters:
   - n_estimators=100: Number of boosting rounds.
   - max_depth=3: Maximum depth of a tree.
   - learning_rate=0.1: Step size shrinkage to prevent overfitting.
   - subsample=0.8: Subsample ratio of the training instance.
   - eval_metric='logloss': Evaluation metric for validation data.
   - use_label_encoder=False: Suppresses a deprecation warning.
   The model is then fit to the scaled training data (X_train_scaled, y_train).

7. **Model Evaluation:** The trained model's performance is evaluated on the test set (X_test_scaled, y_test) using:
   - accuracy_score: Overall accuracy of the predictions.
   - classification_report: Provides precision, recall, and F1-score for each class (0 and 1), offering a comprehensive view of the model's performance, especially for imbalanced datasets.

8. **Model Persistence:** The trained XGBoost model and the fitted StandardScaler are saved using joblib as xgboost_heart_failure_model.pkl and scaler.pkl, respectively. This allows for easy loading and deployment of the model without retraining.

## Deployment
The trained machine learning model is deployed as a web service using a Flask application, making it accessible for real-time predictions via a simple web interface.

### Deployment Steps:
1. **Flask Application (app.py):**
   - A Flask application is initialized.
   - The pre-trained xgboost_heart_failure_model.pkl and scaler.pkl are loaded using joblib when the application starts.

2. **Home Route (/):**
   - Serves the index.html file, which contains the user interface for data input.

3. **Prediction Route (/predict):**
   - Accepts POST requests with patient data in JSON format.
   - Extracts individual features from the incoming JSON payload.
   - Re-computes the same engineered features (creatinine_sodium_ratio, cpk_platelet_ratio, anaemia_diabetes, bp_smoking, age_ejection_interaction) as used during training.
   - Assembles all features into a NumPy array, ensuring the order matches the training data.
   - Scales the input features using the loaded scaler.
   - Uses the loaded model to make a prediction (0 for low risk, 1 for high risk).
   - Returns the prediction as a JSON response.

4. **Web Interface (index.html):**
   - A responsive HTML form is provided for users to input patient clinical data.
   - Uses Tailwind CSS for styling to ensure a clean and modern look.
   - JavaScript handles the form submission:
     - Collects data from the input fields.
     - Sends the data to the Flask /predict endpoint using an asynchronous fetch request.
     - Displays the prediction result (High Risk or Low Risk of Heart Failure) dynamically on the page, with appropriate styling based on the outcome.
     - Includes error handling for network or API issues.

5. **Running the Application:**
   - Ensure app.py, index.html (in a templates folder), xgboost_heart_failure_model.pkl, and scaler.pkl are in the correct directory structure.
   - The Flask application can be run locally using python app.py.
   - It will typically be accessible at http://127.0.0.1:5000/.
  
Below is what out prediction app looks like

![Heart risk prediction](https://github.com/DavieObi/Heart-Failure-Prediction/raw/f5c55882371a43082ccaeafed60f728739fc7883/Heart%20risk%20pred.jpg)

![High risk prediction 2](https://github.com/DavieObi/Heart-Failure-Prediction/raw/f5c55882371a43082ccaeafed60f728739fc7883/High%20risk%20pred%202.jpg)


This deployment setup creates a functional web application that allows users to interact with your trained heart failure prediction model, providing a practical way to utilize the machine learning solution.

