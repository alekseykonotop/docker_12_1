FROM python:latest

WORKDIR /usr/src/my_app

COPY requirements.txt ./app_requirements/requirements.txt
RUN pip install --no-cache-dir -r ./app_requirements/requirements.txt

COPY ./src/ .

CMD  python ./server.py 