from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
#from sklearn.preprocessing import RobustScaler

app = Flask(__name__)


# Load the saved model
with open('fraudDetection.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

#X = data_df[['MerchantName', 'TransactionResult', 'CardIssuer', 'CardCVV', 'TransactionCode', 'TransactionStatus',
#  'CardLimit', 'TransactionDate', 'TransactionFrequency', 'CardVerified']].values

    # Extract the input features
    input_features = [float(data['MerchantName']), float(data['TransactionResult']),float(data['CardIssuer']), float(data['CardCVV']), float(data['TransactionCode']), float(data['TransactionStatus']),
                      float(data['CardLimit']), float(data['TransactionDate']), float(data['TransactionFrequency']), float(data['CardVerified'])]
    

    # Perform any necessary preprocessing on the input features

    #robust_scaler = RobustScaler()

    # # Load the training data for fitting the scaler
    # training_data = pd.read_csv('creditcard.csv')  # Replace with the actual path to your training data

    # # Fit the scaler with the training data
    # robust_scaler.fit(training_data[['V3', 'V9', 'V10', 'V12', 'V14', 'V16', 'V17', 'V2', 'V4', 'V11']])  # Adjust the feature names as per your training data

    # # Transform the input features using the fitted scaler
    # input_features = robust_scaler.transform([input_features])

    # # Reshape the input features to 2-dimensional array
    # input_features = input_features.reshape(1, -1)

    # Make predictions using the loaded model
    predictions = loaded_model.predict([input_features])

     # Apply the if-else condition on the predictions
    if predictions[0] == 0:
        result = 'not fraudulent ✅'
        
    else:
        result = 'FRAUDULENT 💀'

    # Return the result as a JSON response
    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(debug=True)


