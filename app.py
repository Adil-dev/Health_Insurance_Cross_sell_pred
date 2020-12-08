from flask import Flask, request
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.preprocessing import MinMaxScaler
from prep import prep_func


app=Flask(__name__)
Swagger(app)

pickle_in = open("clf.pkl","rb")
clf = pickle.load(pickle_in)
pickle_in_under = open("clf_unuder.pkl", "rb")
clf_under = pickle.load(pickle_in_under)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Health Insurance.
    '> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0,
    'Male' : 0, 'Female' : 1,
    'Yes' : 1, 'No' : 0,

    ---
    parameters:  
      - name: Age
        in: query
        type: number
        required: true
      - name: Vintage
        in: query
        type: number
        required: true
      - name: Gender
        in: query
        type: number
        required: true
      - name: Previously_Insured
        in: query
        type: number
        required: true
      - name: Vehicle_Age
        in: query
        type: number
        required: true
      - name: Vehicle_Damage
        in: query
        type: number
        required: true
      - name: Driving_License
        in: query
        type: number
        required: true
      - name: Policy_Sales_Channel
        in: query
        type: number
        required: true
      - name: Region_Code
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    Age=request.args.get("Age")
    Vintage=request.args.get("Vintage")
    Gender=request.args.get("Gender")
    Previously_Insured=request.args.get("Previously_Insured")
    Vehicle_Age=request.args.get("Vehicle_Age")
    Vehicle_Damage=request.args.get("Vehicle_Damage")
    Driving_License=request.args.get("Driving_License")
    Policy_Sales_Channel=request.args.get("Policy_Sales_Channel")
    Region_Code=request.args.get("Region_Code")
    prediction=clf.predict([[Age,Vintage,Gender,Previously_Insured,Vehicle_Age,Vehicle_Damage,Driving_License,Policy_Sales_Channel,Region_Code]])
    print(prediction)
    return "Hello The answer is"+str(prediction)


@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Upload the csv file.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """

    df = pd.read_csv(request.files.get("file"), sep=',', index_col=['id'])
    prepprop = prep_func(df)
    prediction = clf.predict(prepprop)
    return str(list(prediction))


@app.route('/predict_file_under_sample',methods=["POST"])
def predict_note_file_undersample():
    """Upload the csv file.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """

    df = pd.read_csv(request.files.get("file"), sep=',', index_col=['id'])
    prepprop_under = prep_func(df)
    prediction_under = clf_under.predict(prepprop_under)
    
    return str(list(prediction_under))


if __name__=='__main__':
    app.run(port=8000)