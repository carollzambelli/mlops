FROM python:3.12

RUN mkdir -p /usr

WORKDIR /usr

COPY . /usr

RUN pip install -r requirements.txt

ENV PYTHONPATH="./"

CMD [ "python", "src/predict.py" ]