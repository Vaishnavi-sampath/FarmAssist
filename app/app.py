from flask import Flask, request, redirect, render_template, Markup
from flask import url_for
import pickle


# Importing essential libraries and modules

import numpy as np
#import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.crop import crop_image_dic
import requests
import config
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9


# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']


disease_model_path = 'C:/Users/vaish/Desktop/capstone/FarmAssist/app/models/disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Loading fertilizer prediction model
fertilizer_model_path = 'models/SVM_Fertilizer.pkl'
fertilizer_model = pickle.load(
    open(fertilizer_model_path, 'rb'))

#importing pickle files
fertilizer_target_path = 'models/Ferti_Target.pkl'
fertilizer_target = pickle.load(
    open(fertilizer_target_path, 'rb'))

#----------------------weather function--------------
def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None



def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction
#-----------------------flask app---------------------

app = Flask(__name__)

@app.route('/')
def home():
    title = 'FarmAssist - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop')
def crop_recommend():
    title = 'FarmAssist - Crop Prediction'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'FarmAssist - Fertilizer Recommendation'

    return render_template('ferti.html', title=title)

#-----------------------render prediction page---------------------------


@ app.route('/crop-prediction', methods=['POST'])
def crop_prediction():
    title = 'FarmAssist - Crop Prediction'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            # Get image file name for the predicted crop
            image_file_name = crop_image_dic.get(final_prediction.lower(), 'default.jpg')


            return render_template('croppredict.html', prediction=final_prediction,
                                   image_url=url_for('static', filename='frontend/images/crop_img/' + image_file_name),
                                     title=title)

        else:

            return render_template('try_again.html', title=title)
        



@app.route('/fertilizer-recommendation',methods=['POST'])
def fertilizer_predict():
    title = 'FarmAssist - Fertilizer Recommendation'
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')
    input = [int(temp),int(humi),int(mois),int(soil),int(crop),int(nitro),int(pota),int(phosp)]

    res = fertilizer_target.classes_[fertilizer_model.predict([input])]
    res_scalar = res.item()
    desc=fertilizer_dic[res_scalar]

    return render_template('fertipredict.html',x = ('Predicted Fertilizer is {}'.format(res)),desc=desc,title=title)


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'FarmAssist - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('diseasepredict.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)






if __name__ == "__main__":
    app.run(debug=True)