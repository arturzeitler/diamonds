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

TOPICS = {
    0: 'Business',
    1: 'Entertainment',
    2: 'Politics',
    3: 'Sport',
    4: 'Tech'
}

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
        if isinstance(posted['input'], str) == False:
            log.info('Provided input is indeed of type String')
            return {'status': '"input" key does not return string'}, HTTPStatus.BAD_REQUEST
        jsn = json.dumps({
            'signature_name': 'serving_default',
            'inputs': posted,
        })
        response = requests.post(SERVER_URL, data=jsn)
        log.info("POST request sent")
        response.raise_for_status()
        preds = response.json()['outputs'][0]
        max_val = max(preds)
        max_index = preds.index(max_val)
        log.info("Prediction " + TOPICS[max_index])
        return {'status': TOPICS[max_index]}, HTTPStatus.OK
