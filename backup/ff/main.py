from flask import Flask, render_template, request, jsonify
from gtts import gTTS
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sr
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

dataset = pd.read_csv('cpdata.csv')


le= LabelEncoder()
dataset['label']= le.fit_transform(dataset['label'])

# plt.figure(figsize = (10,6))
# sns.heatmap(dataset.corr(), vmax = 0.9, square = True)
# plt.title("crop attribute correlation")
# plt.show()

# Separate features (X) and labels (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
# sc= StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

#SVM
clf3= SVC(kernel="rbf")
clf3.fit(x_train, y_train)
pred_y1 = clf3.predict(x_test)
print(f"Confusion matrix:\n {confusion_matrix(y_test, pred_y1)}")
print("SVM Accuracy: " + str(accuracy_score(y_test, pred_y1)))

# decision tree classifier
clf1 = DecisionTreeClassifier()
clf1.fit(x_train, y_train)
pred_y1 = clf1.predict(x_test)
print(f"Confusion matrix:\n {confusion_matrix(y_test, pred_y1)}")
print("Desicion tree Accuracy: " + str(accuracy_score(y_test, pred_y1)))

# Naive bayes
clf4= GaussianNB()
clf4.fit(x_train, y_train)
pred_y1= clf4.predict(x_test)
print(f"Confusion matrix:\n {confusion_matrix(y_test, pred_y1)}")
print("Naive Bayes Accuracy: " + str(accuracy_score(y_test, pred_y1)))

# random forest classifier
clf2= RandomForestClassifier()
clf2.fit(x_train, y_train)
pred_y = clf2.predict(x_test)
print(f"Confusion matrix:\n {confusion_matrix(y_test, pred_y)}")
print("Random Forest Accuracy: " + str(accuracy_score(y_test, pred_y)))




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
        predicted_crop = clf2.predict(user_input)[0]

        return render_template('result.html', crop=predicted_crop)

    return render_template('index.html')


# Initialize the recognizer
recognizer = sr.Recognizer()


# Function to capture and transcribe speech
def capture_speech():
    try:
        # Use the microphone to capture the user's voice
        with sr.Microphone() as source:
            bot_message = "Please speak."
            print(bot_message)
            tts = gTTS(text=bot_message, lang='en')
            # tts.save("bot_message.mp3")
            # os.system("mpg321 bot_message.mp3")  # Use mpg321 for Linux, or other players for other platforms
            # os.remove("bot_message.mp3")

            audio = recognizer.listen(source)

        # Recognize the spoken value
        spoken_value = recognizer.recognize_google(audio)

        return spoken_value

    except sr.UnknownValueError:
        bot_message = "Sorry, I could not understand the audio."
        print(bot_message)
        tts = gTTS(text=bot_message, lang='en')
        # tts.save("bot_message.mp3")
        # os.system("mpg321 bot_message.mp3")
        # os.remove("bot_message.mp3")
        return bot_message
    except sr.RequestError as e:
        bot_message = f"Speech recognition request failed: {e}"
        print(bot_message)
        tts = gTTS(text=bot_message, lang='en')
        # tts.save("bot_message.mp3")
        # os.system("mpg321 bot_message.mp3")
        # os.remove("bot_message.mp3")
        return bot_message


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture_speech', methods=['POST'])
def capture_and_fill():
    spoken_value = capture_speech()
    return jsonify({'spokenValue': spoken_value})


if __name__ == '__main__':
    app.run(debug=True)
