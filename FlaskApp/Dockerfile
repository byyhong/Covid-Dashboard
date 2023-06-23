FROM python:3.8
WORKDIR /FlaskApp
COPY FlaskApp/requirements.txt FlaskApp/server2.py daily_covid_death_model.h5 ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

WORKDIR ../data

COPY daily_covid_data.csv ./
  
RUN 
EXPOSE 5000
CMD ["flask", "--app", "server2", "run", "--no-debugger"]

