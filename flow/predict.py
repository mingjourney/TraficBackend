import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input, LSTM, Attention, Concatenate

scaler = joblib.load('./model/total_flow_in_scaler.model')
model = joblib.load('./model/total_flow_in.model')

def set_features(df):
    df['-1day'] = df['flow'].shift(1)
    df['-4day'] = df['flow'].shift(4)
    df['-1week'] = df['flow'].shift(7)
    df['-2week'] = df['flow'].shift(14)
    df['-3week'] = df['flow'].shift(21)
    df['-4week'] = df['flow'].shift(28)
    df['-1day minus -29day'] = df['flow'].shift(1) - df['flow'].shift(29)
    return df

def predict(df, is_workday):
    date = df.index[len(df) - 1] + timedelta(days=1)
    df['is_workday'] = False
    df.loc[date] = {
        'flow': 0,
        'is_workday': is_workday
    }
    df_features = set_features(df.copy())
    df_features = df_features.loc[[date]]
    df_features.drop(columns='flow', inplace=True)
    X = scaler.transform(df_features)
    y = model.predict(X)
    y[0] = round(y[0])
    df.loc[date, 'flow'] = y[0]
    return y[0]

def predict_n(n, df, is_workday):
    y = []
    for i in range(n):
        y.append(predict(df, is_workday[i]))
    return y

stations = pd.read_csv('station.csv')['station_name'].to_list()
stations.sort()

station_flow_in_P = np.load('./model/station_flow_in_P.npy')

def init_model_station_flow_in():
    model = load_model('./model/station_flow_in.h5')
    return model

model_station_flow_in = init_model_station_flow_in()

def predict_station_flow_in(df):
    length = len(df.columns)
    date = df.columns[length - 1] + timedelta(days=1)
    A = np.array(df[df.columns[length - 28: length]])
    k = 64
    P = station_flow_in_P
    Q = (np.linalg.pinv(P) @ A).T
    predict_q = model_station_flow_in.predict(Q.reshape((1, 28, k)))
    predict_a = station_flow_in_P @ predict_q.T
    predict_a = predict_a.T[0]
    predict_a = np.where(predict_a < 0, 0, predict_a)
    predict_a = np.around(predict_a)
    df[date] = predict_a
    return dict(zip(stations, predict_a.tolist()))

def predict_station_flow_in_n(n, df):
    for i in range(n):
        predict_station_flow_in(df)
    length = len(df.columns)
    df = df[df.columns[length - n: length]]
    df.index = stations
    return df

station_flow_out_P = np.load('./model/station_flow_out_P.npy')

def init_model_station_flow_out():
    model = load_model('./model/station_flow_out.h5')
    return model

model_station_flow_out = init_model_station_flow_out()

def predict_station_flow_out(df):
    length = len(df.columns)
    date = df.columns[length - 1] + timedelta(days=1)
    A = np.array(df[df.columns[length - 28: length]])
    k = 64
    P = station_flow_out_P
    Q = (np.linalg.pinv(P) @ A).T
    predict_q = model_station_flow_out.predict(Q.reshape((1, 28, k)))
    predict_a = station_flow_out_P @ predict_q.T
    predict_a = predict_a.T[0]
    predict_a = np.where(predict_a < 0, 0, predict_a)
    predict_a = np.around(predict_a)
    df[date] = predict_a
    return dict(zip(stations, predict_a.tolist()))

def predict_station_flow_out_n(n, df):
    for i in range(n):
        predict_station_flow_out(df)
    length = len(df.columns)
    df = df[df.columns[length - n: length]]
    df.index = stations
    return df

section_flow_up_P = np.load('./model/section_flow_up_P.npy')

def init_model_section_flow_up():
    model = load_model('./model/section_flow_up.h5')
    return model

model_section_flow_up = init_model_section_flow_up()

def predict_section_flow_up(A):  
    k = 64
    P = section_flow_up_P
    Q = (np.linalg.pinv(P) @ A).T
    predict_q = model_section_flow_up.predict(Q.reshape((1, 28, k)))
    predict_a = section_flow_up_P @ predict_q.T
    predict_a = predict_a.T[0]
    predict_a = np.where(predict_a < 0, 0, predict_a)
    predict_a = np.around(predict_a)
    return predict_a

section_flow_down_P = np.load('./model/section_flow_down_P.npy')

def init_model_section_flow_down():
    model = load_model('./model/section_flow_down.h5')
    return model

model_section_flow_down = init_model_section_flow_down()

def predict_section_flow_down(A):
    k = 64
    P = section_flow_down_P
    Q = (np.linalg.pinv(P) @ A).T
    predict_q = model_section_flow_down.predict(Q.reshape((1, 28, k)))
    predict_a = section_flow_down_P @ predict_q.T
    predict_a = predict_a.T[0]
    predict_a = np.where(predict_a < 0, 0, predict_a)
    predict_a = np.around(predict_a)
    return predict_a

peak_flow_morning_in_P = np.load('./model/peak_flow_morning_in_P.npy')

def init_model_peak_flow_morning_in():
    model = load_model('./model/peak_flow_morning_in.h5')
    return model

model_peak_flow_morning_in = init_model_peak_flow_morning_in()

def predict_peak_flow_morning_in(df):
    length = len(df.columns)
    date = df.columns[length - 1] + timedelta(days=1)
    A = np.array(df[df.columns[length - 28: length]])
    k = 64
    P = peak_flow_morning_in_P
    Q = (np.linalg.pinv(P) @ A).T
    predict_q = model_peak_flow_morning_in.predict(Q.reshape((1, 28, k)))
    predict_a = peak_flow_morning_in_P @ predict_q.T
    predict_a = predict_a.T[0]
    predict_a = np.where(predict_a < 0, 0, predict_a)
    predict_a = np.around(predict_a)
    df[date] = predict_a
    return dict(zip(stations, predict_a.tolist()))
    
peak_flow_morning_out_P = np.load('./model/peak_flow_morning_out_P.npy')

def init_model_peak_flow_morning_out():
    model = load_model('./model/peak_flow_morning_out.h5')
    return model

model_peak_flow_morning_out = init_model_peak_flow_morning_out()

def predict_peak_flow_morning_out(df):
    length = len(df.columns)
    date = df.columns[length - 1] + timedelta(days=1)
    A = np.array(df[df.columns[length - 28: length]])
    k = 64
    P = peak_flow_morning_out_P
    Q = (np.linalg.pinv(P) @ A).T
    predict_q = model_peak_flow_morning_out.predict(Q.reshape((1, 28, k)))
    predict_a = peak_flow_morning_out_P @ predict_q.T
    predict_a = predict_a.T[0]
    predict_a = np.where(predict_a < 0, 0, predict_a)
    predict_a = np.around(predict_a)
    df[date] = predict_a
    return dict(zip(stations, predict_a.tolist()))
    
peak_flow_evening_in_P = np.load('./model/peak_flow_evening_in_P.npy')

def init_model_peak_flow_evening_in():
    model = load_model('./model/peak_flow_evening_in.h5')
    return model

model_peak_flow_evening_in = init_model_peak_flow_evening_in()

def predict_peak_flow_evening_in(df):
    length = len(df.columns)
    date = df.columns[length - 1] + timedelta(days=1)
    A = np.array(df[df.columns[length - 28: length]])
    k = 64
    P = peak_flow_evening_in_P
    Q = (np.linalg.pinv(P) @ A).T
    predict_q = model_peak_flow_evening_in.predict(Q.reshape((1, 28, k)))
    predict_a = peak_flow_evening_in_P @ predict_q.T
    predict_a = predict_a.T[0]
    predict_a = np.where(predict_a < 0, 0, predict_a)
    predict_a = np.around(predict_a)
    df[date] = predict_a
    return dict(zip(stations, predict_a.tolist()))
    
peak_flow_evening_out_P = np.load('./model/peak_flow_evening_out_P.npy')

def init_model_peak_flow_evening_out():
    model = load_model('./model/peak_flow_evening_out.h5')
    return model

model_peak_flow_evening_out = init_model_peak_flow_evening_out()

def predict_peak_flow_evening_out(df):
    length = len(df.columns)
    date = df.columns[length - 1] + timedelta(days=1)
    A = np.array(df[df.columns[length - 28: length]])
    k = 64
    P = peak_flow_evening_out_P
    Q = (np.linalg.pinv(P) @ A).T
    predict_q = model_peak_flow_evening_out.predict(Q.reshape((1, 28, k)))
    predict_a = peak_flow_evening_out_P @ predict_q.T
    predict_a = predict_a.T[0]
    predict_a = np.where(predict_a < 0, 0, predict_a)
    predict_a = np.around(predict_a)
    df[date] = predict_a
    return dict(zip(stations, predict_a.tolist()))
    