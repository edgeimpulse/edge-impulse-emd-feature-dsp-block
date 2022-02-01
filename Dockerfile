# syntax = docker/dockerfile:experimental
FROM python:3.9-slim-buster

WORKDIR /app

# Python dependencies
COPY . ./
RUN pip3 install --upgrade pip 
RUN pip3 --no-cache-dir install -r requirements-blocks.txt

COPY third_party /third_party

EXPOSE 4446

CMD python3 -u dsp-server.py
