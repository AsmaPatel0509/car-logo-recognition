from flask import Flask, request, render_template, redirect, flash, url_for, send_from_directory, session
import os
from werkzeug.utils import secure_filename
from flask_session import Session
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

UPLOAD_FOLDER = 'F:/PycharmPrograms/taskQuixote/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
sess = Session()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(__name__)
app.secret_key = "abc"

# @app.route('/setsession')
# def sestsession():
#     session['key'] = request.files()

@app.route('/', methods=['get', 'post'])
def loadindex():
    return render_template("/index.html")

def allowedFile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        if request.files:
            img = request.files["image"]
            print("Uploaded image: ", img.filename)

            if img.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowedFile(img.filename):
                filename = secure_filename(img.filename)

                img.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

                print("Image saved")
                result = testModel(img.filename)
                return render_template('output.html', result=result)
            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("/index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filename = 'http://127.0.0.1:5000/uploads/' + filename
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                                       filename)

def testModel(img):
    imagePath = 'F:/PycharmPrograms/taskQuixote/uploads/' + img
    print(imagePath)
    image = cv2.imread(imagePath)
    # orig = image.copy()

    image = cv2.resize(image, (64, 64))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    print("[INFO] loading network...")
    model = load_model("vggModel.model")

    (volkswagen, hyundai, porsche, honda, audi) = model.predict(image)[0]
    print("=================================================")
    print("Prediction: ")
    print("=================================================")
    print("Honda: ", honda)
    print("Volkswagen: ", volkswagen)
    print("Hyundai: ", hyundai)
    print("Porsche: ", porsche)
    print("Audi: ", audi)

    resultList = [ volkswagen, hyundai, honda, porsche, audi]
    resultDict = {
        "volkswagen": volkswagen,
        "hyundai": hyundai,
        "honda": honda,
        "porsche": porsche,
        "audi": audi
    }
    return resultDict

app.run(debug=True)