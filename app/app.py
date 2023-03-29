from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
fertilizer_model_path = 'models/SVM_Fertilizer.pkl'
fertilizer_model = pickle.load(
    open(fertilizer_model_path, 'rb'))

#importing pickle files
fertilizer_target_path = 'models/Ferti_Target.pkl'
fertilizer_target = pickle.load(
    open(fertilizer_target_path, 'rb'))



@app.route('/')
def welcome():
    return render_template('ferti.html')

@app.route('/fertilizer-predict',methods=['POST'])
def fertilizer_predict():
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

    return render_template('fertipredict.html',x = ('Predicted Fertilizer is {}'.format(res)))

if __name__ == "__main__":
    app.run(debug=True)