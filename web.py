#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import tensorflow.compat.v1 as tf
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import base64
import numpy as np
import matplotlib.pyplot as plt
import  datetime
import time
import io
from PIL import Image
# from multilabel import  Multilabel
from eedetector.yoloee import YoloEE
from Multilabel import  MultiLabel
import  cv2
from poetry import Seq2SeqPredictor, is_quatrain

app = Flask(__name__)

# detector_multilable = Multilabel()
detector_multilable = None
# detector_yoloee = YoloEE()
detector_yoloee = None

Multilll = MultiLabel()
predictor = Seq2SeqPredictor()
# Multilll.test("../pairwiseranking/test.jpg")

@app.route('/', methods=['GET', 'POST'])
def home():
    return '<h1>Home</h1>'

@app.route('/signin', methods=['GET'])
def signin_form():
    return '''<form action="/signin" method="post">
              <p><input name="username"></p>
              <p><input name="password" type="password"></p>
              <p><button type="submit">Sign In</button></p>
              </form>'''

@app.route('/signin', methods=['POST'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username']=='admin' and request.form['password']=='password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'


resSets = {}
@app.route('/upload', methods=['Get'])
def post():
    resSets["value"] = 10
    resSets["resurl"] = 'hello zjg!'
    return json.dumps(resSets, ensure_ascii=False)


@app.route('/upload', methods=['POST'])
def upload():
    resSets = {}
    print('ok')
    img = request.form.get("image")
    image_io = io.BytesIO(base64.b64decode(img))
    img = Image.open(image_io)

    image = np.array(img)
    plt.imshow(image)
    time = datetime.datetime.now()
    plt.imsave('./images/{}{}{}{}{}{}.jpg'.format(time.year,time.month,time.day,time.hour,time.minute,time.second),image)
    plt.show()
    if detector_multilable is None:
        resSets["ret"] = 1
        resSets["msg"] = 'the server is closed, because the natapp is expensive. please contact me'
        return json.dumps(resSets, ensure_ascii=False)
    results = detector_multilable.detect(img)
    if len(results)>0:
        resSets["ret"] = 0
        resSets["msg"] = 'ok'
        resSets["result"] = results  #[{'name':'picture','confidence':21},{'name':'girl','confidence':99}]
    else:
        resSets["ret"] = 1
        resSets["msg"] = 'no tag found, please change your pictures'
    return json.dumps(resSets, ensure_ascii=False)

@app.route('/multilabel', methods=['POST'])
def multilabel():
    resSets = {}
    print('ok')
    img = request.form.get("image")
    image_io = io.BytesIO(base64.b64decode(img))
    img = Image.open(image_io)

    image = np.array(img)
    image = cv2.resize(image, (Multilll.image_size, Multilll.image_size))
    image_data = image.astype(np.float32)
    plt.imshow(image)
    time = datetime.datetime.now()
    plt.imsave('./images/{}{}{}{}{}{}.jpg'.format(time.year,time.month,time.day,time.hour,time.minute,time.second),image)
    plt.show()
    if Multilll is None:
        resSets["ret"] = 1
        resSets["msg"] = 'the server is closed, because the natapp is expensive. please contact me'
        return json.dumps(resSets, ensure_ascii=False)
    results = Multilll.predict(image)
    if len(results)>0:
        resSets["ret"] = 0
        resSets["msg"] = 'ok'
        resSets["result"] = results  #[{'name':'picture','confidence':21},{'name':'girl','confidence':99}]
    else:
        resSets["ret"] = 1
        resSets["msg"] = 'no tag found, please change your pictures'
    return json.dumps(resSets, ensure_ascii=False)



@app.route('/image2poetry', methods=['POST'])
def image2poetry():
    resSets = {}
    print('ok')
    img = request.form.get("image")
    image_io = io.BytesIO(base64.b64decode(img))
    img = Image.open(image_io)

    image = np.array(img)
    image = cv2.resize(image, (Multilll.image_size, Multilll.image_size))
    image_data = image.astype(np.float32)
    plt.imshow(image)
    time = datetime.datetime.now()
    plt.imsave('./images/{}{}{}{}{}{}.jpg'.format(time.year,time.month,time.day,time.hour,time.minute,time.second),image)
    plt.show()
    if Multilll is None and predictor is  None:
        resSets["ret"] = 1
        resSets["msg"] = 'the server is closed, because the natapp is expensive. please contact me'
        return json.dumps(resSets, ensure_ascii=False)
    results = Multilll.predict(image)
    poem = []
    if len(results)>0:
        key_words = [i['name'] for i in results]
        while True:
            poem = predictor.predict(key_words)
            if is_quatrain(poem):
                break

    if len(poem)>0:
        for line in poem:
            print(line)
        resSets["ret"] = 0
        resSets["msg"] = 'ok'
        resSets["result"] = poem  #[{'name':'picture','confidence':21},{'name':'girl','confidence':99}]
    else:
        resSets["ret"] = 1
        resSets["msg"] = 'no poetey generated, please change your pictures'
    return json.dumps(resSets, ensure_ascii=False)


if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=2019,
        debug=True
    )