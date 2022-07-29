import os
from flask import Flask, render_template, flash, send_file,abort
from flask import request
from tensorflow import keras
import pickle
from corr_factors import change_doc
from docx import Document

load_model = keras.models.load_model('multi_class_BiGRU')
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/upload", methods=['GET','POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    doc_file=request.files["doc_file"]
    print(doc_file.filename)
    if (".docx" not in doc_file.filename):
        abort(400)
    doc_path="./docs/"+doc_file.filename
    if not os.path.isdir("./docs/"):
        os.mkdir("./docs/")
    doc_file.save(doc_path)
    document=change_doc(doc_path,tokenizer,load_model)
    document.save(doc_path.rsplit(".",1)[0]+"_edit.docx")
    document.save('./docs/'+"edit.docx")
    #for file in request.files.getlist("file"):
    #    print(file)
    #    filename = file.filename
    #    destination = "/".join([target, filename])
    #    print(destination)
    #    file.save(destination)
    document = Document(doc_path)
    return render_template("home.html",prediction=document)

@app.route("/download")
def download():
    #doc_file=request.files["doc_file"]
    #doc_path="./docs/"+doc_file.filename.rsplit(".",1)[0]+"_edit.docx"
    return send_file('./docs/'+"edit.docx",as_attachment=True)

if __name__== '__main__':
    app.run()