from flask import *
import os
from werkzeug.utils import secure_filename
import label_image
from flask import Flask, render_template, request
from bot import chat

def load_image(image):
    text = label_image.main(image)
    return text

app = Flask(__name__)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

 
  
    
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = load_image(file_path)
        result = result.title()
        d = {"brownspot":" → You are Normal. Keep Calm and Make Healthy Life",
	'Mild Demented':" → It may be a starting stage of the Alzheimer's disease consult the doctor asap and get well soon!!.",
        "Moderate Demented":" → Neurodegenerative diseases are incurable and debilitating conditions that result in progressive degeneration and / or death of nerve cells. This causes problems with movement (called ataxias), mental functioning (called dementias) and affect a person's ability to move, speak and breathe[1]. Neurodegenerative disorders impact many families - these disorders are not easy for the individual nor their loved ones.",
        "Non Demented":" → You are perfectly alright according to your MRI scan image hope you are staying good.",
        "Verymild Demented":" →It is a very mild stage of the Alzheimer disease there is a more probability to recover from this disease get treated from the neurologist and get well soon."}
        result = result+d[result]
        #result2 = result+d[result]
        #result = [result]
        #result3 = d[result]        
        print(result)
        #print(result3)
        os.remove(file_path)
        return result
        #return result3
    return None


@app.route("/home")
def home():
    return render_template("index1.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    exit_list = ['exit','see you later','bye','quit','break']
    if userText.lower() in exit_list:
    	return "Bye..Take Care..Chat with you later!!"
    	#break;
    else:
    	return str(chat(userText))
    	
@app.errorhandler(500)
def internal_error(error):
	return "500 error"

@app.errorhandler(404)
def not_found(error):
    return "404 error",404
    

if __name__ == '__main__':
    app.run()