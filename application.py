# Import necessary libraries
from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import StandardScaler
# Import pandas library
import pandas as pd
import numpy as np
import joblib
# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# Load the fitted StandardScaler
scaler = joblib.load('scaler.pkl')

# Define a function to make predictions
def make_predictions(scaler, input_data):
    print ("input data ",input_data)
    
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)

    # Make predictions using the model
    predictions = model.predict(std_data)
    
    return predictions

# Define route for predicting data
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html', form_data={})  # Render the form template with empty form data
    else:
        # Create CustomData object with form data
       
        gender=float(request.form.get('gender'))
        age=float(request.form.get('age'))
        hypertension=float(request.form.get('hypertension'))
        heart_disease=float(request.form.get('heart_disease'))
        ever_married=float(request.form.get('ever_married'))
        work_type=float(request.form.get('work_type'))
        Residence_type=float(request.form.get('Residence_type'))
        avg_glucose_level=float(request.form.get('avg_glucose_level'))
        bmi=float(request.form.get('bmi'))
        smoking_status=float(request.form.get('smoking_status'))
    
        
        data = (gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,
                bmi,smoking_status)

        # Perform prediction
        prediction = make_predictions(scaler, data)
        print("prediction",prediction[0])

        # Translate prediction to human-readable format
        prediction_text = "Yes" if prediction[0] == 1 else "No"
        
        # Pass form data back to the template
        form_data = {
            'gender': float(request.form.get('gender')),
            'age': float(request.form.get('age')),
            'hypertension': float(request.form.get('hypertension')),
            'heart_disease': float(request.form.get('heart_disease')),
            'ever_married': float(request.form.get('ever_married')),
            'work_type': float(request.form.get('work_type')),
            'Residence_type': float(request.form.get('Residence_type')),
            'avg_glucose_level': float(request.form.get('avg_glucose_level')),
            'bmi': float(request.form.get('bmi')),
            'smoking_status': float(request.form.get('smoking_status'))
        }

        # Render the template with prediction result and form data
        return render_template('index.html', prediction=prediction_text, form_data=form_data)
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
