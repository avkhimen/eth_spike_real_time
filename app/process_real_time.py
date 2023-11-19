import requests
import json
from datetime import datetime
import os
from twilio.rest import Client
import time
import pickle
import numpy as np

while True:

    time.sleep(20)

    current_time = datetime.now()
    print(f'Processed {current_time}')
    if (current_time.hour in [0,4,8,12,16,20]) and (current_time.minute == 10):

        print('Checking prices...')

        end = int(time.mktime(datetime.now().timetuple()))

        start = end - 1000 * 4 * 60 * 60

        ethusd_url = f'https://futures.kraken.com/api/charts/v1/spot/PI_ETHUSD/4h?from={start}&to={end}'

        ethusd_candles = requests.get(ethusd_url).json()['candles']

        ethusd_close_prices = []
        timestamps = []
        for item in ethusd_candles:
            timestamps.append(item['time'])
            ethusd_close_prices.append(float(item['close']))

        btcusd_url = f'https://futures.kraken.com/api/charts/v1/spot/PI_XBTUSD/4h?from={start}&to={end}'

        btcusd_candles = requests.get(btcusd_url).json()['candles']

        btcusd_close_prices = []
        timestamps = []
        for item in btcusd_candles:
            timestamps.append(item['time'])
            btcusd_close_prices.append(float(item['close']))

        close_prices = np.divide(np.array(ethusd_close_prices), np.array(btcusd_close_prices)).tolist()

        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)

        close_prices = np.array(close_prices)

        input_features = np.concatenate((close_prices[-90:], close_prices[-90:]/min(close_prices[-90:]), [min(close_prices[-90:])]), axis=0)

        pred = model.predict([input_features])[0]

        print(pred)

        if pred == 'True':

            account_sid = os.environ['ACCOUNT_SID']
            auth_token = os.environ['AUTH_TOKEN']
            client = Client(account_sid, auth_token)

            message = client.messages.create(
                from_=os.environ['TWILIO_PHONE_NUMBER'],
                body='Eth should spike against BTC',
                to='+17809321716'
            )

            # ts = int('1284101485')

            # print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))