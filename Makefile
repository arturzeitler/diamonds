up-local-train:
	docker-compose -f docker-compose.localtrain.yml up --build

up-aws-train:
	docker-compose -f docker-compose.awstrain.yml up -d --build

up-local-serve:
	docker-compose -f docker-compose.localserve.yml up -d --build

up-aws-serve:
	docker-compose -f docker-compose.awsserve.yml up -d --build

tests:
	docker-compose -f docker-compose.localserve.yml run --rm --no-deps --entrypoint=pytest api /tests/

create-vpc:
	aws cloudformation create-stack --stack-name vpc --template-body file://cf-public-private-vpc.yml --parameters ParameterKey=EnvironmentName,ParameterValue=production --on-failure DO_NOTHING

delete-vpc:
	aws cloudformation delete-stack --stack-name vpc

create-ecscluster:
	aws cloudformation create-stack --stack-name ecs-cluster --template-body file://cf-ecs-cluster.yml --parameters ParameterKey=EnvironmentName,ParameterValue=production --on-failure DO_NOTHING --capabilities CAPABILITY_IAM

delete-ecscluster:
	aws cloudformation delete-stack --stack-name ecs-cluster

create-train-task:
	aws cloudformation create-stack --stack-name ecs-train --template-body file://task-fargate-train.yml --parameters ParameterKey=EnvironmentName,ParameterValue=production --on-failure DO_NOTHING --capabilities CAPABILITY_NAMED_IAM

delete-train-task:
	aws cloudformation delete-stack --stack-name ecs-train

create-alb-public:
	aws cloudformation create-stack --stack-name alb-public --template-body file://cf-alb-external.yml --parameters ParameterKey=EnvironmentName,ParameterValue=production --on-failure DO_NOTHING

delete-alb-public:
	aws cloudformation delete-stack --stack-name alb-public

create-alb-private:
	aws cloudformation create-stack --stack-name alb-private --template-body file://cf-alb-internal.yml --parameters ParameterKey=EnvironmentName,ParameterValue=production --on-failure DO_NOTHING

delete-alb-private:
	aws cloudformation delete-stack --stack-name alb-private

create-api-service:
	aws cloudformation create-stack --stack-name ecs-api --template-body file://service-fargate-api.yml --parameters ParameterKey=EnvironmentName,ParameterValue=production --on-failure DO_NOTHING

delete-api-service:
	aws cloudformation delete-stack --stack-name ecs-api

create-tf-service:
	aws cloudformation create-stack --stack-name ecs-tf-serve --template-body file://service-fargate-tf.yml --parameters ParameterKey=EnvironmentName,ParameterValue=production --on-failure DO_NOTHING

delete-tf-service:
	aws cloudformation delete-stack --stack-name ecs-tf-serve

post-aws-api:
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
"z": [5.55]}]}' -X POST http://alb-p-Publi-4QW7LRNCVV99-2035073095.us-east-2.elb.amazonaws.com/predict

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
