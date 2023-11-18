import requests
import json
from datetime import datetime

url = 'https://futures.kraken.com/api/charts/v1/spot/PI_XBTUSD/4h?from=1629038028&to=1700294134'

r_js = requests.get(url).json()

print(r_js)

# ts = int('1284101485')

# print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))