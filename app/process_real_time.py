import requests
import json
from datetime import datetime
import os

url = 'https://futures.kraken.com/api/charts/v1/spot/PI_XBTUSD/4h?from=1629038028&to=1700294134'

r_js = requests.get(url).json()

print(r_js)

print(r_js.keys())

from twilio.rest import Client

account_sid = os.environ['ACCOUNT_SID']
auth_token = os.environ['AUTH_TOKEN']
client = Client(account_sid, auth_token)

message = client.messages.create(
    from_=os.environ['TWILIO_PHONE_NUMBER'],
    body='Hello there',
    to='+17809321716'
)

print(message.sid)

# ts = int('1284101485')

# print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))