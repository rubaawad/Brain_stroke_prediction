from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('best_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the fitted StandardScaler
try:
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")

def make_predictions(scaler, input_data, feature_names):
    # Check if input_data has the correct format and length
    if not isinstance(input_data, (tuple, list)):
        raise ValueError("Input data must be a tuple or a list.")
    if len(input_data) != len(feature_names):
        raise ValueError(f"Input data must contain {len(feature_names)} features.")
    
    # Convert input data to a numpy array and create a DataFrame with feature names
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)

    # Standardize the input data
    std_data = scaler.transform(input_data_df)

    # Debugging: Print the standardized data
    print("Standardized data:", std_data)

    # Check if the model has the predict method
    if not hasattr(model, 'predict'):
        raise AttributeError("Loaded model does not have a predict method.")
    
    # Make predictions using the model
    predictions = model.predict(std_data)
    
    return predictions

# Define route for predicting data
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html', form_data={})  # Render the form template with empty form data
    else:
        # Retrieve form data
        gender = float(request.form.get('gender'))
        age = float(request.form.get('age'))
        hypertension = float(request.form.get('hypertension'))
        heart_disease = float(request.form.get('heart_disease'))
        ever_married = float(request.form.get('ever_married'))
        work_type = float(request.form.get('work_type'))
        Residence_type = float(request.form.get('Residence_type'))
        avg_glucose_level = float(request.form.get('avg_glucose_level'))
        bmi = float(request.form.get('bmi'))
        smoking_status = float(request.form.get('smoking_status'))
    
        data = (gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)

        features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                    'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

        # Perform prediction
        try:
            prediction = make_predictions(scaler, data, features)
            print("Prediction:", prediction[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', error=f"Error during prediction: {e}", form_data=request.form)

        # Translate prediction to human-readable format
        prediction_text = "Yes" if prediction[0] == 1 else "No"

        # Pass form data back to the template
        form_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        # Render the template with prediction result and form data
        return render_template('index.html', prediction=prediction_text, form_data=form_data)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
