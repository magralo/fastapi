FROM python:3.8

WORKDIR /app

COPY / ./

RUN pip install fastapi uvicorn

EXPOSE 5000  

ENV PORT 5000

CMD uvicorn main:app --reload --port 5000