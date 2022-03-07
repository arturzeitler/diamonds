#FROM python:3.8-slim
FROM public.ecr.aws/a3h9g1a2/python-3.8-slim-buster:latest

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
  && apt-get clean

# add and install requirements
COPY . /
WORKDIR /
RUN python -m pip install --upgrade pip

RUN pip install -r /requirements.txt
RUN chmod +x /gunicorn_starter.sh
ENTRYPOINT ["/gunicorn_starter.sh"]