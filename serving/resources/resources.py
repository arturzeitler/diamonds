import os
import logging
import json
from http import HTTPStatus
from typing import Type
import requests

from flask import request
from flask_restful import Resource

log = logging.getLogger(__name__)

SERVER_URL = os.environ['TF_URL']

class Health(Resource):

    def get(self):
        log.info("GET request received")
        return {'status': 'good'}, HTTPStatus.OK

class Predict(Resource):

    def post(self):
        log.info("POST request received")
        if request.json == None:
            return {'status': 'No JSON provided'}, HTTPStatus.BAD_REQUEST
        posted = request.json
        log.info(posted)
        try:
            if posted['input'] != None:
                log.info('input key found in json')
            else:
                raise KeyError
        except KeyError:
            return {'status': 'No "input" key in JSON'}, HTTPStatus.BAD_REQUEST
        for item in posted['input']:
            if (list(item) == ['carat', 'clarity', 'color', 'cut', 'depth', 'table', 'x', 'y', 'z']) == False:
                return {"status": "Wrong keys in at least one of the inputs"}, HTTPStatus.BAD_REQUEST
        jsn = json.dumps({
            "signature_name": "serving_default",
            "instances": posted['input'],
        })
        response = requests.post(SERVER_URL, data=jsn)
        log.info("POST request sent")
        response.raise_for_status()
        preds = response.json()['predictions']
        log.info("POST response received")
        return {'predictions': preds}, HTTPStatus.OK
