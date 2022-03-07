import json
from http import HTTPStatus

def test_get(client):
    response = client.get('/')
    assert response.status_code == 200

def test_input_keys(client):
    response = client.post('/predict', data=json.dumps(
        {"input": [{"carat": [1.41],
            "clarity": ["I1"],
            "color": ["H"],
            "cut": ["Fair"],
            "depth": [64.7],
            "table": [58.0],
            "x": [7.05],
            "y": [7.0],
            "z": [4.55]},
            {"carat": [1.23],
            "clarity": ["VS2"],
            "color": ["H"],
            "WRONG": ["Ideal"],
            "depth": [66.8],
            "table": [59.0],
            "x": [7.95],
            "y": [6.0],
            "z": [5.55]}]}),
        content_type='application/json')
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json == {"status": "Wrong keys in at least one of the inputs"}

def test_wrong_key(client):
    response = client.post('/predict', data=json.dumps(
        {"wrong_key": [{"carat": [1.41],
            "clarity": ["I1"],
            "color": ["H"],
            "cut": ["Fair"],
            "depth": [64.7],
            "table": [58.0],
            "x": [7.05],
            "y": [7.0],
            "z": [4.55]},
            {"carat": [1.23],
            "clarity": ["VS2"],
            "color": ["H"],
            "cut": ["Ideal"],
            "depth": [66.8],
            "table": [59.0],
            "x": [7.95],
            "y": [6.0],
            "z": [5.55]}]}),
            content_type='application/json')
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json == {"status": 'No "input" key in JSON'}
