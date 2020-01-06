#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request

import json
import os
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


app = Flask(__name__)

# detector_multilable = Multilabel()
detector_multilable = None
# detector_yoloee = YoloEE()
detector_yoloee = None


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
        detections["ret"] = 1
        detections["msg"] = 'the server is closed, because the natapp is expensive. please contact me'
        return json.dumps(detections, ensure_ascii=False)
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
    image = cv2.resize(image, (MultiLabel.image_size, MultiLabel.image_size))
    image_data = image.astype(np.float32)
    plt.imshow(image)
    time = datetime.datetime.now()
    plt.imsave('./images/{}{}{}{}{}{}.jpg'.format(time.year,time.month,time.day,time.hour,time.minute,time.second),image)
    plt.show()
    if Multilll is None:
        detections["ret"] = 1
        detections["msg"] = 'the server is closed, because the natapp is expensive. please contact me'
        return json.dumps(detections, ensure_ascii=False)
    results = Multilll.predict(image)
    if len(results)>0:
        resSets["ret"] = 0
        resSets["msg"] = 'ok'
        resSets["result"] = results  #[{'name':'picture','confidence':21},{'name':'girl','confidence':99}]
    else:
        resSets["ret"] = 1
        resSets["msg"] = 'no tag found, please change your pictures'
    return json.dumps(resSets, ensure_ascii=False)


detections = {}
@app.route('/eedetector', methods=['POST'])
def eedetector():
    detections = {}

    t = time.time()
    img = request.form.get("image")
    name  = request.form.get("name")
    passwd = request.form.get("passwd")
    print('ok')
    image_io = io.BytesIO(base64.b64decode(img))
    img = Image.open(image_io)
    print("第一阶段耗时：{}".format(time.time() - t))
    image = np.array(img)
    plt.imshow(image)
    plt.show()
    dt = datetime.datetime.now()
    plt.imsave('./images/{}{}{}{}{}{}.jpg'.format(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second),image)

    # plt.show()
    if name =='lechatelia' and passwd=="19970327":
        t = time.time()
        if detector_yoloee is None:
            detections["ret"] = 1
            detections["msg"] = 'The server is closed, because the natapp is expensive. please contact me, and I will start the server for you.'
            return json.dumps(detections, ensure_ascii=False)
        img, results = detector_yoloee.detect(img)
        print("检测耗时：{}".format(time.time()-t))
        plt.imshow(img)
        plt.show()
        if len(results)>0:
            detections["ret"] = 0
            detections["msg"] = 'ok'
            detections["result"] = results  #[{'name':'picture','confidence':21},{'name':'girl','confidence':99}]
            img = Image.fromarray(img)
            outbuffrt = io.BytesIO()
            img.save(outbuffrt, format='JPEG')
            bytedata = outbuffrt.getvalue()
            detections["image"] = bytes.decode(base64.b64encode(bytedata))
            #str(base64.b64encode(str.encode(str(img))), encoding ='utf-8)
        else:
            detections["ret"] = 1
            detections["msg"] = 'no foreign object found, please change your pictures'

    else:
        detections["ret"] = 1
        detections["msg"] = 'your user info is wrong'

    return json.dumps(detections, ensure_ascii=False)

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=2019,
        debug=True
    )