# Customer Churn Prediction Project

This project demonstrates how to build a customer churn prediction model using a neural network implemented in TensorFlow (Keras). The workflow includes data preprocessing, model building, training, evaluation, and deployment using a Streamlit web application.

## Overview

**Problem:** Predict whether a customer will leave a bank (churn) based on customer features.

**Data:** This example uses the "Churn_Modelling.csv" dataset, which contains customer information including demographics, account details, and whether they exited the bank.

**Key Steps:**
1. **Data Preprocessing:**  
   - Dropping irrelevant columns  
   - Encoding categorical variables (both label encoding and one-hot encoding)  
   - Splitting data into train and test sets  
   - Scaling features using `StandardScaler`

2. **Model Architecture (ANN):**  
   - A simple feed-forward neural network built using Keras:
     - Input layer
     - Hidden Layers: Two hidden layers (64 units and 32 units) with ReLU activation
     - Output Layer: Single neuron with sigmoid activation for binary classification

3. **Training the Model:**  
   - Compiling the model with `Adam` optimizer and `BinaryCrossentropy` loss  
   - Implementing `EarlyStopping` to prevent overfitting
   - Logging metrics with `TensorBoard` for visualization

4. **Evaluation:**  
   - Evaluating model performance on the test set  
   - Reporting accuracy  
   - Inspecting predictions

5. **Deployment using Streamlit:**  
   - A simple web app to input customer details  
   - The app uses saved encoders and scalers to preprocess user input and then predict churn probability

## Project Structure

```
.
├─ Churn_Modelling.csv         # Dataset
├─ model.h5                    # Saved Keras model
├─ label_encoder_gender.pkl     # Saved LabelEncoder for Gender
├─ onehot_encoder_geo.pkl       # Saved OneHotEncoder for Geography
├─ scaler.pkl                   # Saved StandardScaler
├─ app.py                      # Streamlit app script
└─ README.md                   # This README file
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- streamlit

To install the required packages, run:

```bash
pip install -r requirements.txt
```

*(Create a `requirements.txt` file if needed. It should include the packages used.)*

## Steps to Run

1. **Data Preparation & Model Training:**
   - Ensure `Churn_Modelling.csv` is in the same directory.
   - Run the Jupyter notebook or Python script (not included as a `.py` file in this snippet but can be extracted from the code provided) that performs the following:
     - Load and preprocess data
     - Train the model
     - Save the encoders, scaler, and model to disk
   - After training, you should have `model.h5`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, and `scaler.pkl` saved.

2. **Start the Streamlit App:**
   - Ensure you have all saved model files in the same directory as `app.py`.
   - Run the following command:
     ```bash
     streamlit run app.py
     ```
   - This will start a local web server and open the Streamlit UI in your browser.

3. **Predict Churn:**
   - Fill in the customer details in the Streamlit app’s sidebar.
   - Click "Predict" (if implemented, or it might auto-predict on input change).
   - The app will display the probability of churn and a message indicating whether the customer is likely to churn.

## Files Explained

- **Churn_Modelling.csv**: Original dataset.
- **app.py**: Streamlit application code. Renders a UI for user inputs and predicts churn using the trained model.
- **model.h5**: Trained Keras model weights and architecture.
- **label_encoder_gender.pkl**: Trained `LabelEncoder` for the Gender column.
- **onehot_encoder_geo.pkl**: Trained `OneHotEncoder` for the Geography column.
- **scaler.pkl**: Trained `StandardScaler` for numerical features.

## Model Performance

- The performance can be evaluated through accuracy on the test dataset.
- The code snippet shows how to print the final test accuracy. Further metrics (precision, recall, F1-score, AUC) can be computed as needed.

## Customization

- Modify the architecture of the neural network as needed.
- Adjust hyperparameters (e.g., learning rate, batch size, epochs) to improve performance.
- Add additional metrics or visualization tools (e.g., confusion matrix, ROC curve).

## Acknowledgments

- Dataset: The dataset used is a popular example dataset for churn modeling and is widely available online.
- Tools: TensorFlow, Keras, scikit-learn, Pandas, Numpy, Streamlit.
