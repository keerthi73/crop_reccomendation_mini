# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Create a Flask web application
app = Flask(__name__)

# Load the training dataset (replace 'your_dataset.csv' with the actual dataset file path)
dataset = pd.read_csv('cpdata.csv')

# Separate features (X) and labels (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Create and train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Define a route to display the form
# ... (previous code)


# Define a route to display the form
@app.route('/', methods=['GET', 'POST'])
def predict_crop():
    if request.method == 'POST':
        # Get user inputs from the form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Make a prediction using the trained model
        user_input = [[temperature, humidity, ph, rainfall]]
        predicted_crop = clf.predict(user_input)[0]

        return render_template('result.html', crop=predicted_crop)

    return render_template('index.html')

# ... (rest of the code)


if __name__ == '__main__':
    app.run(debug=True)
