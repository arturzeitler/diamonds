up-local-train:
	docker-compose -f docker-compose.localtrain.yml up --build

up-local-serve:
	docker-compose -f docker-compose.localserve.yml up --build

tests:
	docker-compose -f docker-compose.localserve.yml run --rm --no-deps --entrypoint=pytest api /tests/

post-local-api:
	curl -H 'Content-Type: application/json' -d '{"input": [{"carat": [1.41], \
"clarity": ["I1"], \
"color": ["H"], \
"cut": ["Fair"], \
"depth": [64.7], \
"table": [58.0], \
"x": [7.05], \
"y": [7.0], \
"z": [4.55]}, \
{"carat": [1.23], \
"clarity": ["VS2"], \
"color": ["H"], \
"cut": ["Ideal"], \
"depth": [66.8], \
"table": [59.0], \
"x": [7.95], \
"y": [6.0], \
"z": [5.55]}]}' -X POST http://localhost:5001/predict

post-local:
	curl -d '{"instances":[{"carat": [1.41], \
"clarity": ["I1"], \
"color": ["H"], \
"cut": ["Fair"], \
"depth": [64.7], \
"table": [58.0], \
"x": [7.05], \
"y": [7.0], \
"z": [4.55]}, \
{"carat": [1.23], \
"clarity": ["VS2"], \
"color": ["H"], \
"cut": ["Ideal"], \
"depth": [66.8], \
"table": [59.0], \
"x": [7.95], \
"y": [6.0], \
"z": [5.55]}]}' -X POST http://localhost:8501/v1/models/diamonds:predict
