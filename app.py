from flask import Flask, request, jsonify
from functions import extract_nail
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json


# load model
json_file = open('best_bn_model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights('best_bn_model.h5')

# app
app = Flask(__name__)


@app.route('/prediction/', methods=['GET', 'POST'])
def predict():

    image = request.args.get('image')

    img = cv2.imread(image)
    cropped = extract_nail(img)

    # in case something went wrong in the cropping, we just use the full image
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        cropped = img[200:200+800, 550:550+850]
        resized = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    else:
        resized = cv2.resize(cropped, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

    
    prediction = model.predict(resized.reshape(1, 128, 128, 1))
    prob = np.max(prediction)
    prediction = np.argmax(prediction)

    if prediction == 0:
        return jsonify("This is a bad nail. Probability: {:.4f}".format(prob))
    elif prediction == 1:
        return jsonify("This is a good nail. Probability: {:.4f}".format(prob))


if __name__ == "__main__":
    app.run('127.0.0.1')