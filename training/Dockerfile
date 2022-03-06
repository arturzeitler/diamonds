FROM python:3.8-slim
COPY ./ /
WORKDIR /
RUN python -m pip install --upgrade pip

RUN pip install -r /requirements.txt
RUN pip install -e ./dia_train

RUN mkdir /checkpoints
RUN mkdir -p /diamonds/0001

CMD ["/run.py"]
ENTRYPOINT ["python"]
