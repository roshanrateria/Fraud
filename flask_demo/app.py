from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load pre-trained models and encoders
with open('is_fraud.pkl', 'rb') as f:
    is_fraud_model = pickle.load(f)

with open('fraud_type.pkl', 'rb') as f:
    fraud_type_model = pickle.load(f)

# Load label encoders for features
encoders = {}
for feature in ['card_type', 'location', 'purchase_category', 'time_of_day']:
    with open(f'{feature}.pkl', 'rb') as f:
        encoders[feature] = pickle.load(f)

with open('fraud_type_le.pkl', 'rb') as f:
    fraud_type_encoder = pickle.load(f)

# Helper function to plot and encode plots as base64
def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Define the prediction function
def predict_fraud(amount, card_type, location, purchase_category, customer_age, time_of_day):
    # Preprocess input
    input_data = pd.DataFrame({
        'amount': [amount],
        'card_type': [card_type],
        'location': [location],
        'purchase_category': [purchase_category],
        'customer_age': [customer_age],
        'time_of_day': [time_of_day]
    })

    # Label encode the inputs
    for feature in ['card_type', 'location', 'purchase_category', 'time_of_day']:
        input_data[feature] = encoders[feature].transform(input_data[feature])

    # Fraud Detection (Binary Classification)
    is_fraud_pred_prob = is_fraud_model.predict_proba(input_data)[0]
    is_fraud_class = np.argmax(is_fraud_pred_prob)

    # Fraud Type Classification (Multiclass Classification)
    fraud_type_pred_prob = fraud_type_model.predict_proba(input_data)[0]
    fraud_type_class = fraud_type_encoder.inverse_transform([np.argmax(fraud_type_pred_prob)])[0]

    # Bar charts for probabilities
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Fraud Detection Probabilities
    axes[0].bar(['Not Fraudulent', 'Fraudulent'], is_fraud_pred_prob)
    axes[0].set_title('Fraud Detection Probability')
    axes[0].set_ylabel('Probability')

    if is_fraud_class == 1:
        # Plot Fraud Type Probabilities
        fraud_types = fraud_type_encoder.classes_
        axes[1].bar(fraud_types, fraud_type_pred_prob)
        axes[1].set_title('Fraud Type Probability')
        axes[1].set_ylabel('Probability')
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plot_base64 = plot_to_base64(fig)

    return f"{max(is_fraud_pred_prob)*100:.2f}% Confidence: {'Fraudulent' if is_fraud_class == 1 else 'Not Fraudulent'}", fraud_type_class if is_fraud_class == 1 else "", plot_base64

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        amount = float(request.form['amount'])
        card_type = request.form['card_type']
        location = request.form['location']
        purchase_category = request.form['purchase_category']
        customer_age = int(request.form['customer_age'])
        time_of_day = request.form['time_of_day']

        # Get prediction results
        fraud_result, fraud_type, plot_base64 = predict_fraud(
            amount, card_type, location, purchase_category, customer_age, time_of_day
        )

        return render_template('result.html', 
                               fraud_result=fraud_result, 
                               fraud_type=fraud_type, 
                               plot_base64=plot_base64)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
