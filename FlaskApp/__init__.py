import dash
import dash_core_components as dcc
import dash_html_components as html
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
from datetime import date, timedelta
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
for i in range(28,208):
    if(j%4!=0):
        list_index.append(i)
    j+=1
# # load dataset
#dataset = pd.read_csv("/var/www/FlaskApp/data/Dec17.csv")
dataset = pd.read_csv("/var/www/FlaskApp/data/update_data.csv")
#df.set_index("test_date", inplace = True)
dataset.set_index("test_date",inplace = True)
dataset[dataset < 0] = 0
values = dataset.values
#df.set_index("test_date", inplace = True)
#dataset.set_index("Test Date",inplace = True)
# integer encode direction
encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])

# # ensure all data is float
values = values.astype('float32')
# # normalize features
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_x = scaler_x.fit_transform(values[:,:3])
scaled_y = scaler_y.fit_transform(values[:,None,-1])
scaled = concatenate((scaled_x,scaled_y),axis = 1)
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
#sys.path.insert(0,'/var/www/FlaskApp/FlaskApp')

app = dash.Dash(__name__)
#model = load_model('/var/www/FlaskApp/FlaskApp/dec_model.h5')
model = load_model('/var/www/FlaskApp/FlaskApp/non_negative_dec_model.h5')
#model = load_model('/var/www/FlaskApp/FlaskApp/non_negative_data_dec_model.h5')

#result = []
#month_result = []
#for i in range(45):
#    month_result.append([])
def get_result(local_reframe, x_days_back):
    result = []
    for i in range(x_days_back, x_days_back + 45):
        row_restriction = local_reframe.shape[0]-45+i
        column_restriction = local_reframe.shape[1]-i*4
        test_data = local_reframe.iloc[row_restriction:,column_restriction-28:column_restriction].values
        yhat = model.predict(test_data.reshape((test_data.shape[0], n_days, n_features)))
        inv_yhat = scaler_y.inverse_transform(yhat)
        row = 0
        column = 44
        while row < inv_yhat.shape[0] and column < inv_yhat.shape[1]:
            result.append(inv_yhat[row][column])
            row+=1
            column-=1
    return result

def get_monthly_result(local_reframe, x_days_back):
    month_result = []
    for i in range(45):
        month_result.append([])
    for i in range(x_days_back, x_days_back + 45):
        row_restriction = local_reframe.shape[0]-45+i
        column_restriction = local_reframe.shape[1]-i*4
        test_data = local_reframe.iloc[row_restriction:,column_restriction-28:column_restriction].values
        yhat = model.predict(test_data.reshape((test_data.shape[0], n_days, n_features)))
        inv_yhat = scaler_y.inverse_transform(yhat)
        prediction_date = 0
        while(prediction_date < 45 - i):
            row = prediction_date
            column = 44
            while (row < inv_yhat.shape[0] and column < inv_yhat.shape[1]):
                month_result[prediction_date].append(inv_yhat[row][column])
                row+=1
                column-=1
            prediction_date+=1
    return month_result

def prediction(local_reframe, x_days_back):
    return get_monthly_result(local_reframe, x_days_back), get_result(local_reframe, x_days_back)

month_result, result = prediction(reframed_copy, 0)
year = dataset.iloc[dataset.shape[0]-1:dataset.shape[0] ,: ].index.tolist()[0][:4]
month = dataset.iloc[dataset.shape[0]-1:dataset.shape[0] ,: ].index.tolist()[0][5:7]
day = dataset.iloc[dataset.shape[0]-1:dataset.shape[0] ,: ].index.tolist()[0][8:10]

#Future 45 days prediction along 
#real_45_day_death = []
#real_45_day_death_x = []
#count = 0
#for i in range(dataset.shape[0], dataset.shape[0]+45):
#    real_45_day_death.append(int(dataset.iloc[i]['Death Cases']))
#    count+=1
#    real_45_day_death_x.append((count))
forty_five_prediction_fig = go.Figure()

max_date = date(int(year),int(month.lstrip('0')) ,int(day.lstrip('0')))
for i in range(0,45):
  curr_date = str(max_date + timedelta(int(i)+1))
  curr_date_transform = str(int(curr_date[5:7].lstrip('0'))) +'/'+ str(int(curr_date[8:].lstrip('0')))  +'/' +str(int(curr_date[:4]))
  forty_five_prediction_fig.add_trace(go.Box(y =  month_result[i], name = curr_date_transform))
#forty_five_prediction_fig.add_trace(go.Scatter( x = real_45_day_death_x, y = real_45_day_death, name = '12/18/2020 - 01/31/2021 '))

#Today's death cases prediction along with real death cases
#real_death = int(dataset.iloc[dataset.shape[0]]['Death Cases'])

result_df = pd.DataFrame(result, columns = ['Predicted Death Cases'])

tomorrow = str(date(int(year),int(month.lstrip('0')) ,int(day.lstrip('0'))) + timedelta(1))
tomorrow_reformed = str(int(tomorrow[5:7].lstrip('0'))) +'/'+ str(int(tomorrow[8:].lstrip('0')))  +'/' +str(int(tomorrow[:4]) )

result_x = [tomorrow_reformed for i in range(len(result))]
result_df['date'] = result_x
single_prediction_fig = px.box(result_df, x="date", y="Predicted Death Cases", points="all")
#single_prediction_fig.add_trace(go.Scatter(x = [tomorrow_reformed], y = [real_death], name = tomorrow_reformed  ))

#result_df.to_csv('/var/www/FlaskApp/data/tomo_pred.csv')

single_prediction_fig.update_layout(
	autosize = True,
	width = 400,
	height = 300,
	plot_bgcolor='rgb(25, 25, 112)',
	paper_bgcolor = 'rgb(25, 25, 112)',
	font_family="Courier New",
    	font_color="white",
)

forty_five_prediction_fig.update_layout(
        autosize = True,
        height = 700,
        width = 1000,
	plot_bgcolor='rgb(25, 25, 112)',
        paper_bgcolor = 'rgb(25, 25, 112)',
        font_family="Courier New",
        font_color="white",
	xaxis_title='Date',
	yaxis_title='Death Cases',
)





app.layout = html.Div(children=[
    html.Div(children =[       
            html.Div(children = [
               html.H2(children ='Future Predicted Death Cases Trend '),
               dcc.Graph(id='45 day prediction result', figure=forty_five_prediction_fig),
            ],style={'display': 'inline-block', 'width' : '70%', 'float':'left','height':'100%', 'textAlign':'center'} ),
	    html.Div(children = [
		html.H4(children = 'Future Predicted Death Cases'),
                dcc.Graph(id='selected death cases'),
                dcc.DatePickerSingle(
                    id = 'pick future date',
                    min_date_allowed=date(int(year),int(month.lstrip('0')) ,int(day.lstrip('0'))) + timedelta(1),
                    max_date_allowed=date(int(year),int(month.lstrip('0')) ,int(day.lstrip('0'))) + timedelta(45),
                    initial_visible_month=date(int(year),int( month.lstrip('0')) ,int(day.lstrip('0'))) + timedelta(1),
                    date=date(int(year),int( month.lstrip('0')) ,int(day.lstrip('0'))) + timedelta(1), 
                 ),
           	 
	    	 html.H4(children = str(date(int(year),int(month.lstrip('0')) ,int(day.lstrip('0'))) + timedelta(1)) + ' predicted death cases'),
                 dcc.Graph(id='Dec 18 predict death', figure=single_prediction_fig),
	   ], style={'display': 'inline-block','border-radius': '15px','float':'right', "plot_bgcolor": "rgba(25,25,112)", 'textAlign':'center'}), 
    ], style = {"plot_bgcolor": "rgba(0,0,255,0.5)"}),
])


@app.callback(
    Output('selected death cases', 'figure'),
    Input('pick future date', 'date'),
)
def update_fig(days):
    fig = go.Figure()
    picked_date = date(int(days[:4]), int(days[5:7].lstrip('0')), int(days[8:].lstrip('0')))    
    max_date = date(int(year),int(month.lstrip('0')) ,int(day.lstrip('0')))
    str_date =str(int(days[5:7].lstrip('0'))) +'/'+ str(int(days[8:].lstrip('0')))  +'/' +str(int(days[:4]) )
    fig.add_trace(go.Box(y = month_result[(picked_date - max_date).days-1], name = str_date + " death cases" ))
    fig.update_layout(
   	autosize = True,
	width  = 400,
	height = 300,	
    	plot_bgcolor='rgb(25, 25, 112)',
        paper_bgcolor = 'rgb(25, 25, 112)',
        font_family="Courier New",
        font_color="white", 
    )
    return fig


server = app.server
if __name__ == '__main__':
    app.run_server(debug=True)
