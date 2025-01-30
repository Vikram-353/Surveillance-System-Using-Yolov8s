import os
from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import pyttsx3

app = Flask(__name__)

# Initialize the YOLO model
model = YOLO(r"C:\Users\Dell\Downloads\Survallence System\models\HumanActivuty\model\best.pt")

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  
        self.engine.setProperty('volume', 0.9)

    def convert_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Create a new instance of TextToSpeech for each POST request
        text_to_speech = TextToSpeech()

        imagefile = request.files['imagefile']
        image_path = "../../../images/" + imagefile.filename
        imagefile.save(image_path)
        image = cv2.imread(image_path)

        # Perform prediction
        result = model(image)
        prob = result[0].probs.top1conf
        prob1 = prob.item() * 100

        if prob1 >= 90:
            # Get predicted class name
            names_dict = result[0].names
            index_of_max_prob = result[0].probs.top1
            predicted_class_name = names_dict[index_of_max_prob]

            # Convert image to base64 format for rendering in HTML
            _, img_encoded = cv2.imencode('.png', image)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            # Convert predicted class name to speech
            text_to_speech.convert_to_speech(predicted_class_name)

            return render_template('index.html', prediction=predicted_class_name, img_base64=img_base64)
        else:
            msg = "Can not predict. Probability is too low: {:.2f}".format(prob1)
            text_to_speech.convert_to_speech(msg)
            return render_template('index.html', msg=msg)
    else:
        return render_template('index.html', prediction=None, img_base64=None)

if __name__ == '__main__':
    app.run(port=3001, debug=True)
