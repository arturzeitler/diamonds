import logging
import sys

from flask import Flask
from flask_restful import Api
from flask_cors import CORS

from resources.resources import Health, Predict

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

log.info("Logger created")

def create_app():
    """Initialize the core application."""
    app = Flask(__name__)
    api = Api(app)
    CORS(app)
    api.add_resource(Health, '/')
    api.add_resource(Predict, '/predict')
    log.info("create_app run")
    return app
