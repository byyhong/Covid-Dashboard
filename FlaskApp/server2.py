import json

import dash
import flask
from dash import html
from dash import dcc
import pandas as pd
import sys
import os
import statistics
from keras.models import load_model
import numpy as np
from math import sqrt
import numpy as np
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sys
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import plotly.express as px
import statistics
import math
from keras.models import load_model
from dash.dependencies import Input, Output, State
from datetime import date, timedelta, datetime
import mysql.connector
import test
from flask import Flask
from dateutil.parser import parse
from datetime import date
from flask_cors import CORS


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):

        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


j = 1
list_index = []
for i in range(28, 208):
    if (j % 4 != 0):
        list_index.append(i)
    j += 1
# # load dataset

#Server path
# dataset = pd.read_csv("/var/www/FlaskApp/data/Dec17.csv")
# dataset = pd.read_csv("/var/www/FlaskApp/data/update_data.csv")

#Local path
dataset = pd.read_csv("./data/daily_covid_data.csv")

# df.set_index("test_date", inplace = True)
dataset.set_index("test_date", inplace=True)
dataset.loc[dataset['Death Cases']<0,'Death Cases']=0
values = dataset.values
# df.set_index("test_date", inplace = True)
# dataset.set_index("Test Date",inplace = True)
# integer encode direction
encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])

# # ensure all data is float
values = values.astype('float32')
# # normalize features
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_x = scaler_x.fit_transform(values[:, :3])
scaled_y = scaler_y.fit_transform(values[:, None, -1])
scaled = concatenate((scaled_x, scaled_y), axis=1)
# # specify the number of lag hours
n_days = 7
n_features = 4
# frame as supervised learning
reframed = series_to_supervised(scaled, n_days, 45)
reframed_copy = reframed.copy()
reframed.drop(reframed.columns[list_index], axis=1, inplace=True)

# split into train and test sets
v = reframed.values
value = v.copy()
n_train_time = int(reframed.shape[0] * 0.8)
train = value[:n_train_time, :]
test = value[n_train_time:, :]
# split into input and outputs
n_obs = n_days * n_features
train_X, train_y = train[:, :n_obs], train[:, -45:]
test_X, test_y = test[:, :n_obs], test[:, -45:]

# # reshape input to be 3D [samples, timesteps, features]
train_x = train_X.reshape((train_X.shape[0], n_days, n_features))
test_x = test_X.reshape((test_X.shape[0], n_days, n_features))
# sys.path.insert(0,'/var/www/FlaskApp/FlaskApp')

app = dash.Dash(__name__)
#Server path
# model = load_model('/var/www/FlaskApp/FlaskApp/dec_model.h5')
model = load_model('../FlaskApp/daily_covid_death_model.h5')

# model = load_model('/var/www/FlaskApp/FlaskApp/non_negative_data_dec_model.h5')

# result = []
# month_result = []
# for i in range(45):
#    month_result.append([])

def get_result(local_reframe, x_days_back):
    result = []
    for i in range(x_days_back, x_days_back + 45):
        row_restriction = local_reframe.shape[0] - 45 + i
        column_restriction = local_reframe.shape[1] - i * 4
        test_data = local_reframe.iloc[row_restriction:, column_restriction - 28:column_restriction].values
        yhat = model.predict(test_data.reshape((test_data.shape[0], n_days, n_features)))
        inv_yhat = scaler_y.inverse_transform(yhat)
        row = 0
        column = 44
        while row < inv_yhat.shape[0] and column < inv_yhat.shape[1]:
            result.append(inv_yhat[row][column])
            row += 1
            column -= 1
    return result


def get_monthly_result(local_reframe, x_days_back):
    month_result = []
    for i in range(45):
        month_result.append([])
    for i in range(x_days_back, x_days_back + 45):
        row_restriction = local_reframe.shape[0] - 45 + i
        column_restriction = local_reframe.shape[1] - i * 4
        test_data = local_reframe.iloc[row_restriction:, column_restriction - 28:column_restriction].values
        yhat = model.predict(test_data.reshape((test_data.shape[0], n_days, n_features)))
        inv_yhat = scaler_y.inverse_transform(yhat)
        prediction_date = 0
        while (prediction_date < 45 - i):
            row = prediction_date
            column = 44
            while (row < inv_yhat.shape[0] and column < inv_yhat.shape[1]):
                month_result[prediction_date].append(inv_yhat[row][column])
                row += 1
                column -= 1
            prediction_date += 1
    return month_result


def prediction(local_reframe, x_days_back):
    return get_monthly_result(local_reframe, x_days_back), get_result(local_reframe, x_days_back)

month_result, result = prediction(reframed_copy, 0)

config = {
    'host' : 'localhost',
    'user' : 'root',
    'database' : 'Covid_Dash',
    'password' : 'Hlbj513851@',
}

mydb = mysql.connector.connect(**config)
my_cursor = mydb.cursor(buffered=True)
my_cursor.execute("CREATE DATABASE if not exists Covid_Dash")
my_cursor.execute("USE Covid_Dash")
death_cases_table = """CREATE TABLE if not exists Death_Case(
                  date DATE,
                  death_cases DOUBLE
);"""
my_cursor.execute(death_cases_table)
year = dataset.iloc[dataset.shape[0] - 1:dataset.shape[0], :].index.tolist()[0][:4]
month = dataset.iloc[dataset.shape[0] - 1:dataset.shape[0], :].index.tolist()[0][5:7]
day = dataset.iloc[dataset.shape[0] - 1:dataset.shape[0], :].index.tolist()[0][8:10]

max_date = date(int(year), int(month.lstrip('0')), int(day.lstrip('0')))
x_axis_date = []
y_axis_death = []
# y_axis_death_std = []
past_100_x_axis_date = []
for i in range(100-2, -2,-1):
    curr_date = str(max_date - timedelta(int(i) + 1))
    curr_date_transform = str(int(curr_date[:4])) + "-" + str(int(curr_date[5:7].lstrip('0'))) + '-' + str(int(curr_date[8:].lstrip('0')))
    past_100_x_axis_date.append(curr_date_transform)
for i in range(0, 45):
    curr_date = str(max_date + timedelta(int(i) + 1))
    curr_date_transform = str(int(curr_date[:4])) + "-" + str(int(curr_date[5:7].lstrip('0'))) + '-' + str(int(curr_date[8:].lstrip('0')))
    x_axis_date.append(curr_date_transform)

for i in range(len(month_result)):
    for j in range(len(month_result[i])):
        check_if_exist = '''SELECT * FROM Death_Case WHERE date=\'{}\' AND death_cases={}
        '''.format(parse(x_axis_date[i]).strftime("%Y-%m-%d") , month_result[i][j])
        my_cursor.execute(check_if_exist)
        output = my_cursor.fetchone()
        if(output==None):
            to_insert = """INSERT ignore INTO Death_Case (date, death_cases)
            VALUES(
                \'{}\',
                {}
            );
            """.format(parse(x_axis_date[i]).strftime("%Y-%m-%d"), month_result[i][j])
            my_cursor.execute(to_insert)
mydb.commit()
my_cursor.close()
mydb.close()
app = Flask(__name__)
CORS(app)
@app.route('/prediction', methods = ['GET'])
def prediction():
    mydb = mysql.connector.connect(**config)
    my_cursor = mydb.cursor(buffered=True)
    my_cursor.execute("SELECT * FROM Death_Case")
    all_data = my_cursor.fetchall()
    to_return = [list(row) for row in all_data]
    for row in to_return:
        row[0] = row[0].strftime("%Y-%m-%d")
    return to_return




@app.route('/prediction/<string:date_param>', methods = ['GET'])
def prediction_by_date(date_param):
    print(date_param)
    mydb = mysql.connector.connect(**config)
    my_cursor = mydb.cursor(buffered=True)
    date_param = datetime.strptime(date_param, '%Y-%m-%d')
    my_cursor.execute("SELECT * FROM Death_Case WHERE date = \'{}\'".format(date_param))
    all_data = my_cursor.fetchall()
    to_return = [list(row) for row in all_data]
    for row in to_return:
        row[0] = row[0].strftime("%Y-%m-%d")
    return to_return







