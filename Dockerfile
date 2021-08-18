FROM python:3.8


ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
RUN pip install fastapi
RUN pip install gunicorn 
RUN pip install uvicorn 
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_lg


CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 main:app --timeout 240