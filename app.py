from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("model/exam_score_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    test_prep = request.form['test_preparation_course']
    parental_education = request.form['parental_level_of_education']
    reading_score = float(request.form['reading_score'])

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'test preparation course': [test_prep],
        'parental level of education': [parental_education],
        'reading score': [reading_score]
    })

    # Predict the math score
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Math Score: {prediction:.2f}')

if __name__ == "__main__":
    print("Starting Flask Server for Student Exam Score Prediction...")
    app.run(debug=True)