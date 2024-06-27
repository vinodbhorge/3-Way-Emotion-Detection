import secrets,os,sys,librosa,glob,flask
import tensorflow as tf
from playsound import playsound
import sqlite3

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# CUDA_VISIBLE_DEVICES=""

import pandas as pd
import tkinter
from tkinter import *
from flask import render_template,url_for,flash,redirect,request,abort
from emotions import app,db,bcrypt,login_manager
from emotions.forms import RegistrationForm,LoginForm,SubmitText
from emotions.models import User
from flask_login import login_user,current_user,logout_user,login_required
from keras.models import load_model,model_from_json
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
#from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2 as cv
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# # config.gpu_options.per_process_gpu_memory_fraction = 0.9
# session = tf.compat.v1.Session(config=config)
from keras.layers import *
from keras.models import Sequential
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
  
mp = {'happy':'https://manybooks.net/categories','sad':'https://www.youtube.com/watch?v=F9wbogYwTVM'}

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
model12 = load_model('Emotion_little_vgg.h5')
model = load_model('emotions\\chatbot_model.h5')



model1 = load_model("emotions\\mood_text\\model_lstm.h5")
#model1 = ktrain.load_predictor('C:\\Users\\COMP\\Desktop\\AI-Therapist-main\\emotions\\mood_text\\tf_model')
#model1 = ktrain.get_predictor('C:\\Users\\COMP\\Desktop\\AI-Therapist-main\\emotions\\mood_text\\tf_model.h5', preproc)
import numpy as np
from keras.models import load_model
##from keras.preprocessing.text import Tokenizer
###from keras_preprocessing.sequence import pad_sequences
##from keras.utils import pad_sequences
#from keras.preprocessing.sequence import pad_sequences
classes12 = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# json_file = open('D:\Machine Learning\EmotionRecognizer\emotions\model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# print("Loaded model from disk")

# lister = ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']
c1 = ['neutral','calm','happy','surprised']

import pickle
modelll=pickle.load(open("new.pkl","rb"))

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
#model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('emotions\\intents.json').read())
words = pickle.load(open('emotions\\words.pkl','rb'))
classes = pickle.load(open('emotions\\classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    print(msg)
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

    
def suggest_remedy(mood):
    if mood in ('Happy','Neutral','Surprise'):
        return redirect_to_remedy(mood)
    else:
        return redirect_to_remedy(mood)

def redirect_to_remedy(mood):
    return render_template(mp.get(mood,'https://www.youtube.com/watch?v=F9wbogYwTVM'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('home.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html',title='About Page')

@app.route('/home')
def home():
    return render_template('home.html',title='Home Page')

@app.route('/login')
def login():
    return render_template('index.html',title='Home Page')
@app.route('/signup')
def signup():
    return render_template('index.html',title='Home Page')
@app.route('/predict')
def predict_mood():
    final_label = None
    cap = cv.VideoCapture(0)
    got = False
    while True:
        ret,frame = cap.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        
        for x,y,w,h in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv.resize(roi_gray,(48,48),interpolation=cv.INTER_AREA)
            
            if(np.sum([roi_gray])!=0):
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                
                preds = model12.predict(roi)[0]
                label = classes12[preds.argmax()]
                label_position = (x,y)
                final_label = label
                # got = True
                # break
                cv.putText(frame,label,label_position,cv.FONT_HERSHEY_COMPLEX,2,(0,255,0))
            else:
                cv.putText(frame,'No Face Found',(20,60),cv.FONT_HERSHEY_COMPLEX,2,(0,0,255))
        # if got:
        #     break
        cv.imshow('Emotion Detector',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    # print("Done")
    print(final_label)
    if final_label in ('Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'):
        return render_template(f"{final_label}.html")
    


    

@app.route('/logout',methods=["GET","POST"])
def logout():
    logout_user()
    return render_template("index.html")

@app.route('/mood_text',methods=['GET','POST'])
def mood_text():
    form=SubmitText()
    if form.validate_on_submit():
        text = str(form.text.data)
        out=modelll.predict([text])
        print(out[0])

        
        print("text entered is \n\n\n\n {} \n\n\n\n ".format(text))
        check=['joy', 'neutral','surprise']   # 'Happy','Neutral','Sad','Surprise'     
        if out[0] == "joy" :
            return render_template("Happy.html")
        elif out[0] == "neutral" :
            return render_template("Neutral.html")
        elif out[0] == "surprise" :
            return render_template("Surprise.html")
        else:
            return render_template("Sad.html")

    return render_template('text_page.html',title='Submit_Text',form=form)
def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

def record_mood(mood):
    if form.validate_on_submit():
        db.session.add(mood)
        db.session.commit()
        flash(f'Your mood has been recorded!','success')
        return redirect(url_for('login'))
    return render_template('register.html',title='Register',form=form)

@app.route('/chatbot')
def talk_to_chatbot():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot_response(userText))

def convert_class_to_emotion(pred):        
    label_conversion = {'0': 'neutral',
                        '1': 'calm',
                        '2': 'happy',
                        '3': 'sad',
                        '4': 'angry',
                        '5': 'fearful',
                        '6': 'disgust',
                        '7': 'surprised'}

    for key, value in label_conversion.items():
        if int(key) == pred:
            label = value
    return label
