
from werkzeug.utils import secure_filename
from flask import Flask,render_template,jsonify,request,send_from_directory
from flask import request
from flask import render_template
import numpy as np
from multiprocessing import Pool
import autoencoderForGan as midi
import GAN as gan
import os




app = Flask(__name__)


############################################api############################################

#index
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/run_auto', methods=['GET', 'POST'])
def run_auto():
    epochs = int(request.form['epochs'])
    batch_size = int(request.form['batch_size'])
    lr = float(request.form['lr'])
    midi.app_run(epochs,batch_size,lr)
    return jsonify({"errno":1, "msg":"success!"})

@app.route('/run_gan', methods=['GET', 'POST'])
def run_gan():
    epochs = int(request.form['epochs'])
    batch_size = int(request.form['batch_size'])
    d_lr = float(request.form['d_lr'])
    g_lr = float(request.form['g_lr'])
    gan.app_run(epochs,batch_size,d_lr,g_lr)
    return jsonify({"errno":1, "msg":"success!"})


@app.route("/download_auto", methods=['GET'])
def download_file_auto():
    filename = "auto_gen_music_auto.mid"
    directory = os.getcwd()  
    return send_from_directory(directory, filename, as_attachment=True)

@app.route("/download_gan", methods=['GET'])
def download_file_gan():
    filename = "auto_gen_music_gan.mid"
    directory = os.getcwd()  
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/upload', methods=['post'])
def upload():
    fname = request.files.get('file')  
    if fname:
        new_fname = r'upload/' + fname.filename
        fname.save(new_fname)  
        return jsonify({"filename":new_fname})
    else:
        return '{"msg": "error"}'


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
