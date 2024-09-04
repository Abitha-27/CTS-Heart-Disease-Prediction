from flask import Flask, request, render_template, redirect, url_for, flash,session
from pymongo import MongoClient
from PIL import Image
import pytesseract
import random
import pickle
import re
import io
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'cts-project-heart-pred'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
"""
home 
login
signup
symptom
user-db creation
dashboard*
report-debug*

"""


with open(r'models/disease_type_classifier.pkl', 'rb') as model_file:
    disease_type_classifier = pickle.load(model_file)

with open(r'models/symptoms_classifier.pkl', 'rb') as symptoms_file:
    symptoms_classifier = pickle.load(symptoms_file)

with open(r'models/symptoms_classifier_label.pkl', 'rb') as encoder_file:
    disease_label = pickle.load(encoder_file)

hrt_atk_model = load_model('models\Myocardial_Infarction_detection_model.h5')
hrt_atk_scaler = joblib.load('models\scaler_Myocardial_Infarction.pkl')
aor_aneu_model = load_model('models\Aortic_aneurysm_detection_model.h5')
aor_aneu_scaler = joblib.load('models\scaler_aortic_aneurysm.pkl')
hyper_ten_model = load_model('models\hypertension_detection_model.h5')
hyper_ten_scaler = joblib.load('models\scaler_hypertension.pkl')
stroke_model = load_model('models\stroke_detection_model.h5')
stroke_scaler = joblib.load('models\scaler_stoke.pkl')

def get_db_connection():
    try:
        client = MongoClient("mongodb+srv://abithas2711:Abitha2003@cluster0.9jcddh7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        db = client['cts']  
        print("Connected to MongoDB Atlas successfully!")
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {e}")
        return None
    
def encode_age(age):
    if 1 <= age <= 5:
        return 1
    elif 6 <= age <= 20:
        return 2
    elif 21 <= age <= 50:
        return 3
    elif age >= 51:
        return 4
    else:
        return 0  # Default or invalid age handling

def check_hrt_atk_values(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)

    # Define the regular expression patterns to find the required values
    patterns = {
        'Troponin_I' : r'Troponin_\|\s+([\d.]+)',
        'Troponin_T' :r'Troponin_T\s+([\d.]+)',
        'SEX': r'Sex\s*:\s*(\w+)'
    }

    # Initialize a dictionary to store the results
    extracted_values = {}

    # Search for each pattern in the extracted text
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1) if key in ['SEX'] else float(match.group(1))
        else:
            extracted_values[key] = None
        # Extract relevant values
        Troponin_I = extracted_values.get('Troponin_I', 0.0)
        Troponin_T = extracted_values.get('Troponin_T', 0.0)
        gender = 1 if extracted_values.get('SEX', '').lower() == 'male' else 0


        # Create the input array
        manual_input = np.array([[gender,Troponin_I,Troponin_T]])

        # Normalize the input using the loaded scaler
        manual_input_scaled = hrt_atk_scaler.transform(manual_input)

        # Predict using the loaded model
        prediction = hrt_atk_model.predict(manual_input_scaled)

        # Interpreting the prediction
        result = "Heart Attack" if prediction[0] > 0.5 else "Healthy"

        return result

def check_aortic_aneurysm_values(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    patterns = {
        'Systolic_BP': r'Systolic_BP\s+(\d+\.?\d*)',
        'Diastolic_BP': r'Diastolic_BP\s+(\d+\.?\d*)',
        'HDL Cholesterol': r'HDL Cholesterol\s+(\d+\.?\d*)',
        'LDL Cholesterol': r'LDL Cholesterol\s+(\d+\.?\d*)',
        'Triglycerides': r'Triglycerides\s+(\d+\.?\d*)',
        'SEX': r'Sex\s*:\s*(\w+)'
    }

    # Initialize a dictionary to store the results
    extracted_values = {}

    # Search for each pattern in the extracted text
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1) if key in ['SEX'] else float(match.group(1))
        else:
            extracted_values[key] = None

    # Extract relevant values
    Systolic_BP = extracted_values.get('Systolic_BP', 0.0)
    Diastolic_BP = extracted_values.get('Diastolic_BP', 0.0)
    hdl_cholesterol = extracted_values.get('HDL Cholesterol', 0.0)        
    ldl_Cholesterol = extracted_values.get('LDL Cholesterol', 0.0)        
    Triglycerides = extracted_values.get('Triglycerides', 0.0)        

    gender = 1 if extracted_values.get('SEX', '').lower() == 'male' else 0

    # Create the input array
    manual_input = np.array([[Systolic_BP,Diastolic_BP,ldl_Cholesterol,hdl_cholesterol,Triglycerides,gender]])

    # Normalize the input using the loaded scaler
    manual_input_scaled = aor_aneu_scaler.transform(manual_input)

    # Predict using the loaded model
    prediction = aor_aneu_model.predict(manual_input_scaled)

    # Interpreting the prediction
    result = "aortic aneurysm Disease" if prediction[0] > 0.5 else "Healthy"
    return result

def check_hypertension_values(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    patterns = {
        'Systolic_BP': r'Systolic_BP\s+([\d.]+)',
        'Diastolic_BP': r'Diastolic_BP\s+([\d.]+)',
        'SEX': r'Sex\s*:\s*(\w+)'
    }
    # Initialize a dictionary to store the results
    extracted_values = {}

    # Search for each pattern in the extracted text
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1) if key in ['SEX'] else float(match.group(1))
        else:
            extracted_values[key] = None
    Systolic_BP = extracted_values.get('Systolic_BP', 0.0)
    Diastolic_BP = extracted_values.get('Diastolic_BP', 0.0)
    gender = 1 if extracted_values.get('SEX', '').lower() == 'male' else 0


    # Create the input array
    manual_input = np.array([[gender,Systolic_BP,Diastolic_BP]])

    # Normalize the input using the loaded scaler
    manual_input_scaled = hyper_ten_scaler.transform(manual_input)

    # Predict using the loaded model
    prediction = hyper_ten_model.predict(manual_input_scaled)

    # Interpreting the prediction
    result = "Stroke Disease Present" if prediction[0] > 0.5 else "Healthy"
    return result

def check_stroke_values(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    patterns = {
        'HDL Cholesterol': r'HDL Cholesterol\s+(\d+\.?\d*)',
        'LDL Cholesterol': r'LDL Cholesterol\s+(\d+\.?\d*)',
        'Glucose Fasting': r'GLUCOSE, FASTING,\s+(\d+\.?\d*)',
        'SEX': r'Sex\s*:\s*(\w+)'
    }

    # Initialize a dictionary to store the results
    extracted_values = {}

    # Search for each pattern in the extracted text
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1) if key in ['SEX'] else float(match.group(1))
        else:
            extracted_values[key] = None
    
    # Extract relevant values
        hdl_cholesterol = extracted_values.get('HDL Cholesterol', 0.0)
        ldl_cholesterol = extracted_values.get('LDL Cholesterol', 0.0)
        fasting_blood_sugar = extracted_values.get('Glucose Fasting', 0.0)
        gender = 1 if extracted_values.get('SEX', '').lower() == 'male' else 0

        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])

        # Create the input array
        manual_input = np.array([[systolic_bp, diastolic_bp, hdl_cholesterol, ldl_cholesterol, fasting_blood_sugar, gender]])

        # Normalize the input using the loaded scaler
        manual_input_scaled = stroke_scaler.transform(manual_input)

        # Predict using the loaded model
        prediction = stroke_model.predict(manual_input_scaled)

        # Interpreting the prediction
        result = "Stroke Disease Present" if prediction[0] > 0.5 else "Healthy"
        return result

db = get_db_connection()
isLogin = False

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login.html')
def render_login():    
    return render_template('login.html')

@app.route('/signup.html')
def render_signup():
    return render_template('signup.html')

@app.route('/signup',methods=['POST'])
def signup():
    role = request.form.get('role')
    username = request.form.get('username')
    email  = request.form.get('email')
    password = request.form.get('password')
    user ={
        'username': username,
        'email':email,
        'password':password
    }

    if db is not None:
        try:
            users_collection = db['labtech']
            if role=='doctor':
                users_collection = db['doctor'] 
            users_collection.insert_one(user)
            flash('signup successful')
            return redirect('/login.html')
        except Exception as e:
            print(e)
            return redirect('/signup.html')
    return redirect('/signup.html')

@app.route('/doctor.html') #lab-tech incomplete
def doctorPage():
    if isLogin:
        return render_template('/doctor.html',username=session.get('username'),email=session.get('email'))
    else:
        # return render_template('/doc4.html',username=username,email=email)
        return render_template('/login.html')

@app.route('/symptom.html',methods=['POST'])
def load_symptomPage():
    global isLogin
    if isLogin:
        PName = request.form.get('patientName')
        PPhoneNo= request.form.get('patientPhoneNo')
        if db is not None:
            try:
                patient_collection = db['patient']
                patient = patient_collection.find_one({'P_Name':PName,'P_PhoneNo':PPhoneNo})

                if patient is not None:
                    print('patient found')
                    session['P_id']=patient['P_id']
                    session['P_Name']=patient['P_Name']   
                else:
                    P_id =random.randint(10000,99999)
                    patient ={
                        'P_id': P_id,
                        'P_Name': PName,
                        'P_PhoneNo':PPhoneNo,
                    }
                    # print('patient created')
                    # print(patient)
                    session['P_id']=patient['P_id']
                    session['P_Name']=patient['P_Name']
                    patient_collection.insert_one(patient)
                return render_template('/symptom.html',P_id=session.get('P_id'),P_Name=session.get('P_Name'))  
            except Exception as e:
                print(e)
                print('error')
                
        return redirect('/signup.html')
    else:
        return redirect('/login.html')

@app.route('/predict_symptoms',methods=['POST'])
def predict_symtoms():
    if isLogin:
        symptoms_order = [
            'Irritability', 'Heart Block/Failure', 'Rash', 'Feeding', 'Crying', 
            'Shortness of Breath', 'Heart Beat', 'Chest Pain', 'Fatigue', 'Sweating',
            'Dizziness', 'Nausea', 'Pain Radiation', 'Fainting', 'Coughing up blood',
            'Coughing up (without blood)', 'Swelling', 'Cyanosis', 'Weight Loss',
            'Fever', 'Bloating', 'Headache', 'Nosebleeding', 'Seizure',
            'Wheezing', 'Bodyache', 'Breathlessness'
        ]

        # Initialize all symptoms array with zeros
        symptoms_array = [0] * len(symptoms_order)

        # Extract vital symptoms from the form
        age = int(request.form.get('age', 0))
        age = encode_age(age)
        symptoms_array[symptoms_order.index('Heart Block/Failure')] = int(request.form.get('Heart Block/Failure', 0))
        symptoms_array[symptoms_order.index('Shortness of Breath')] = int(request.form.get('Shortness of Breath', 0))
        symptoms_array[symptoms_order.index('Chest Pain')] = int(request.form.get('Chest Pain', 0))
        symptoms_array[symptoms_order.index('Fatigue')] = int(request.form.get('Fatigue', 0))
        symptoms_array[symptoms_order.index('Dizziness')] = int(request.form.get('Dizziness', 0))

        # Extract other symptoms from the textarea
        other_symptoms_text = request.form.get('otherSymtoms', '')

        # Use regular expressions to detect symptoms in the textarea
        for i, keyword in enumerate(symptoms_order):
            if keyword not in ['Heart Block/Failure', 'Shortness of Breath', 'Chest Pain', 'Fatigue', 'Dizziness']: # Skip vital symptoms already processed
                if re.search(keyword, other_symptoms_text, re.IGNORECASE):
                    symptoms_array[i] = 1  # Set to 1 if the symptom is found in the textarea input
        
        isolated_symptoms = [
            age, 
            symptoms_array[symptoms_order.index('Irritability')],
            symptoms_array[symptoms_order.index('Feeding')],
            symptoms_array[symptoms_order.index('Shortness of Breath')],
            symptoms_array[symptoms_order.index('Heart Beat')],
            symptoms_array[symptoms_order.index('Chest Pain')],
            symptoms_array[symptoms_order.index('Fatigue')],
            symptoms_array[symptoms_order.index('Sweating')],
            symptoms_array[symptoms_order.index('Dizziness')],
            symptoms_array[symptoms_order.index('Nausea')],
            symptoms_array[symptoms_order.index('Pain Radiation')],
            symptoms_array[symptoms_order.index('Fainting')],
            symptoms_array[symptoms_order.index('Coughing up (without blood)')],
            symptoms_array[symptoms_order.index('Swelling')],
            symptoms_array[symptoms_order.index('Cyanosis')],
            symptoms_array[symptoms_order.index('Fever')],
            symptoms_array[symptoms_order.index('Wheezing')],
            symptoms_array[symptoms_order.index('Bodyache')]
        ]

        disease_type =disease_type_classifier.predict([isolated_symptoms])[0]

        # print(f"disease_type: {disease_type}")

        # Combine age with symptoms data
        data_for_prediction = [age] + symptoms_array
        data_for_prediction.append(disease_type)

        disease_name = symptoms_classifier.predict([data_for_prediction])[0]
        disease_name = disease_label.inverse_transform([disease_name])[0]



        # Perform prediction using the collected data
        # prediction_result = model.predict([data_for_prediction]) # Replace with actual model prediction code

        # For demonstration, we'll just return the data
        return render_template('Dashboard.html', disease_name=disease_name)
    else:
        return redirect('/login.html')
    
@app.route('/labtechnician.html')
def render_labtechnician():
    if isLogin:
        return render_template('/labtechnician.html',username=session.get('username'),email=session.get('email'))
    else:
        return render_template('/login.html')
    
@app.route('/login', methods=['POST'])
def login():
    global isLogin
    # Get data from the submitted form
    role = request.form.get('role')
    username = request.form.get('username')
    password = request.form.get('password')
    # print(role+" "+username+" "+password)
    
    if db is not None:
        try:
            users_collection = db['labtech']
            if role=='doctor':
                users_collection = db['doctor'] 
             # Replace 'users' with your collection name
            user = users_collection.find_one({'username': username, 'password': password})
            print(user)
            if user:
                # print('login ss')
                session['username']=user['username']
                session['email']=user['email']
                
                # print(user)
                # If a matching user is found, login is successful
                flash('Login successful!', 'success')
                # Redirect to a dashboard or home page
                isLogin= True
                if(role=='doctor'):
                    # return render_template('/doctor.html',username=session.get('username'),email=session.get('email'))
                    return redirect('/doctor.html')
                else:
                    # return render_template('/labtechnician.html',username=session.get('username'),email=session.get('email'))
                    return redirect('/labtechnician.html')
                
            else:
                # If no matching user is found, login fails
                # print('login failed')
                flash('Invalid username or password. Please try again.', 'error')
                return redirect(url_for('home'))
        except Exception as e:
            print(f"Error querying MongoDB Atlas: {e}")
            # flash('An error occurred. Please try again later.', 'error')
            # return redirect(url_for('home'))

    # Redirect to the home page after processing
    return redirect(url_for('home')) 
  
@app.route('/predict_report', methods=['POST'])
def predict_report():
    # Get data from the form
    session['P_id']=request.form.get('patient-id')
    session['P_Name']=request.form.get('patientName')
    session['report_type'] = request.form.getlist('report')[0]
    return render_template('/lab_tec_sub.html')

@app.route('/uploadReport', methods=['POST'])
def process_report():
    if 'testFile' not in request.files:
            return redirect('/lab_tec_sub.html')

    image_file = request.files['testFile']
    
    if(session['report_type']=='ar'):
        result = check_aortic_aneurysm_values(image_file)
    elif(session['report_type']=='hrt-atk'):
        result = check_hrt_atk_values(image_file)
    elif(session['report_type']=='hyp-ten'):
        result = check_hypertension_values(image_file)
    elif(session['report_type']=='stroke'):
        result = check_stroke_values(image_file)



    return result
    
    

    # Extract values from the image
        
if __name__ == '__main__':
    app.run(debug=True)


