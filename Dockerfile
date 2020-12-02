FROM python:3
COPY . /usr/app/
EXPOSE 8000
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD python app.py