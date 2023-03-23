from flask import Flask,request,render_template
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
def hello_world():
   return 'Hello World'

@app.route('/demo',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('demo.html')
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
            return render_template('demo.html', img=img,results=results)
        return render_template('demo.html')
        # results = data.predict_food_image(image,model_load)
        # print(results)
        # return render_template('demo.html',results=results)



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        
