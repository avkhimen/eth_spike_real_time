import requests
import json
from datetime import datetime
import os
from twilio.rest import Client
import time
import pickle

end = int(time.mktime(datetime.now().timetuple()))

start = end - 1000 * 4 * 60 * 60

# url = 'https://futures.kraken.com/api/charts/v1/spot/PI_XBTUSD/4h?from=1262332861&to=1700294134'

url = f'https://futures.kraken.com/api/charts/v1/spot/PI_XBTUSD/4h?from={start}&to={end}'

candles = requests.get(url).json()['candles']

close_prices = []
timestamps = []
for item in candles:
    timestamps.append(item['time'])
    close_prices.append(item['close'])

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(close_prices[-90:])

print(pred)

# account_sid = os.environ['ACCOUNT_SID']
# auth_token = os.environ['AUTH_TOKEN']
# client = Client(account_sid, auth_token)

# message = client.messages.create(
#     from_=os.environ['TWILIO_PHONE_NUMBER'],
#     body='Hello there',
#     to='+17809321716'
# )

# print(message.sid)

# ts = int('1284101485')

# print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))