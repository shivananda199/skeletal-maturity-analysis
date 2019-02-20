from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
app = Flask(__name__)

cwd = os.getcwd()

UPLOAD_FLODER = os.path.join(cwd, 'uploads')


app.config['UPLOAD_FLODER'] = UPLOAD_FLODER

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/page1.html')
def page1():
    return render_template("page1.html")

@app.route('/uploader', methods = ['GET', 'POST'] )
def upload():
    if request.method == 'POST' :
        f = request.files['file']
        x = os.path.join(app.config['UPLOAD_FLODER'], f.filename)
        f.save(x)
        return render_template("page1.html")

if __name__ == "__main__":
    app.run(debug=True)