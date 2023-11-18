import pandas as pd

def load_ts(price_type):
    ts = pd.read_csv('data/original_files/XBTUSD_60.csv', header=None,
                    names=['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2'])
    return ts[price_type].tolist()
