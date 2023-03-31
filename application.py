from flask import Flask,request,render_template, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
# import pandas as pd
import tensorflow as tf
import keras 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from sklearn.preprocessing import StandardScaler
# from src.pipeline.predict_pipeline import Predict_pipeline
category={
    0: ['burger','Burger'], 1: ['butter_naan','Butter Naan'], 2: ['chai','Chai'],
    3: ['chapati','Chapati'], 4: ['chole_bhature','Chole Bhature'], 5: ['dal_makhani','Dal Makhani'],
    6: ['dhokla','Dhokla'], 7: ['fried_rice','Fried Rice'], 8: ['idli','Idli'], 9: ['jalegi','Jalebi'],
    10: ['kathi_rolls','Kaathi Rolls'], 11: ['kadai_paneer','Kadai Paneer'], 12: ['kulfi','Kulfi'],
    13: ['masala_dosa','Masala Dosa'], 14: ['momos','Momos'], 15: ['paani_puri','Paani Puri'],
    16: ['pakode','Pakode'], 17: ['pav_bhaji','Pav Bhaji'], 18: ['pizza','Pizza'], 19: ['samosa','Samosa']
}
application=Flask(__name__)
app=application
## Route for a demo page
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

def predict_image(filename,model):
    img_ = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.   
    
    prediction = model.predict(img_processed)
    
    index = np.argmax(prediction)
    return category[index][1]
    
    # plt.title("Prediction - {}".format(category[index][1]))
@app.route('/')
def home():
   return render_template('home.html')
#    templates\x\x.html
#    return render_template("recipes/x.html")



@app.route('/demo',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('demo1.html')
    else:
        if request.method == 'POST':
            file = request.files['img']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD'], filename))
            img = os.path.join(app.config['UPLOAD'], filename)
            # load the model
            Model_path = "notebook\model_v1_inceptionV3.h5"
            model_load = load_model(Model_path)
            # model_load = load_model('artifacts\model_v1_inceptionV3.h5')
            # data=Predict_pipeline()
            # results = data.predict_food_image(img,model_load)
            results = predict_image(img,model_load)
            return render_template('demo1.html', img=img,results=results)
        return render_template('demo1.html')
        # results = data.predict_food_image(image,model_load)
        # print(results)
        # return render_template('demo.html',results=results)
@app.route('/about')
def about():
   return render_template('recipe-pages/about.html')
@app.route('/contact')
def contact():
   return render_template('recipe-pages/contact.html')
@app.route('/recipes')
def recipes():
   return render_template('recipe-pages/recipes.html')

@app.route('/pizza')
def pizza():
   return render_template('recipe-pages/pizza.html')

@app.route('/samosa')
def samosa():
   return render_template('recipe-pages/samosa.html')

@app.route('/masala_chai')
def masala_chai():
   return render_template('recipe-pages/masala_chai.html')

@app.route('/fried_rice')
def fried_rice():
   return render_template('recipe-pages/fried_rice.html')

@app.route('/Dhokla')
def Dhokla():
   return render_template('recipe-pages/Dhokla.html')

@app.route('/dal_makhani')
def dal_makhani():
   return render_template('recipe-pages/dal_makhani.html')

@app.route('/chole_bhature')
def chole_bhature():
   return render_template('recipe-pages/chole_bhature.html')
@app.route('/chapati')
def chapati():
   return render_template('recipe-pages/chapati.html')
@app.route('/butter_naan')
def butter_naan():
   return render_template('recipe-pages/butter_naan.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        
