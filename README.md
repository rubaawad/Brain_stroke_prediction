# Brain Stroke Prediction
    Brain_stroke_prediction is a machine learning project that aims to predict brain storke based on various  features. It utilizes different machine learning algorithms to build predictive models and evaluates their performance using various metrics.


# Installation
    To run the project locally, follow these steps:

    # Clone the repository:
        git clone https://github.com/rubaawad/Brain_stroke_prediction.git
    # Navigate to the project directory:
        cd Brain_stroke_prediction
    # Install the required dependencies:
        pip install -r requirements.txt
# Notebooks
    In src/components/notebook/data you can find these files:

    # EDA Notebook
        EDA.ipynb: Exploratory Data Analysis notebook for analyzing the dataset.
    # Model Training Notebook
        MODEL TRAINING.ipynb: Notebook for training machine learning models.
# Data
    The project uses the Brain_stroke_prediction Dataset from Kaggle, which contains various  features such as['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi','gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']These features are essential in understanding the factors that influence brain stroke. The dataset is preprocessed before training the models.
# Usage
# Exploratory Data Analysis
The EDA.ipynb notebook covers the following steps:

# Importing Required Packages:
Pandas for data manipulation
Numpy for numerical operations
Matplotlib and Seaborn for data visualization
Scipy for statistical functions

# Loading Data:
Load your dataset and perform initial data checks.

#Data Cleaning and Preparation:
Handle missing values, outliers, and data transformations.

# Data Visualization:
Visualize data distributions, relationships, and patterns using various plots.

# Model Training
The MODEL TRAINING.ipynb notebook includes:
# Importing Required Packages:
Pandas, Numpy, Matplotlib, and Seaborn as mentioned above.
# Loading Data:
Load the cleaned dataset from the EDA step.
# Model Building:
Train different machine learning models and evaluate their performance.
# Model Evaluation:
Use metrics to assess the performance of the models and select the best one.
# Models
    The project employs several machine learning models, including:
    Logistic Regression
    Gradient Boosting
    Support Vector Machine
    Decision Tree
    Random Forest
    Voting classifier
# Evaluation
    The performance of each model is evaluated using the following metrics:
    Accuracy
    Precision
    Recall/Sensitivity
    Specificity
    F1-score
    ROC AUC
# Web Application
    The project includes a web application for predicting customer satisfaction using the trained machine learning models. To run the web application:

    Navigate to the templates directory.
    Install Flask if not already installed: pip install Flask.
    Run the Flask application: python application.py.
    Access the web application in your browser at http://localhost:5000/predict.
# Docker
    Usage Instructions:
        Pull the Docker Image:
            docker pull rubamahgoob/stroke_prediction_app:latest

    Run the Docker Container:
        docker run -d -p 5000:5000 rubamahgoob/stroke_prediction_app:latest

    Access the Application:
        Once the container is running, you can access the Flask application by navigating to http://localhost:5000/predict in your web browser.

    Input Health Parameters:
        On the home page of the application, you'll find a form where you can input various health parameters such as'age', 'hypertension', 'heart_disease',etc.

    Get Prediction Results:
        After entering the health parameters and clicking on the "Predict" button, the application will provide a prediction result indicating whether the person will has brain storke or not.

    Note:	
        Make sure Docker is installed and running on your system before pulling and running the Docker image.
        Ensure that port 5000 is not being used by any other application on your system, as it is used by the Flask application to serve the web interface.
        
    You can customize the Docker container's port mapping if port 5000 is already in use on your system. For example, you can map it to a different port using the -p flag in the docker run command.

# Contact
    For any questions or feedback, please contact:

    Ruba Awad
    Email: rubaabdalla44@gmail.com