# from binance.um_futures import UMFutures
import pandas as pd
from time import sleep
from time import time
# from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta


# from binance.error import ClientError

# client = UMFutures()

# from binance.client import Client

# Binance.US API

# client = Client(api_key, api_secret, tld='us', testnet=False)


# def get_tickers_usdt():
#     try:
#         tickers = []
#         resp = client.ticker_price()
#         for elem in resp:
#             if 'USDT' in elem['symbol']:
#                 tickers.append(elem['symbol'])
#         return tickers
#     except ClientError as error:
#         print(
#             f"Found error. status: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}")


intervals = {'1m': 1,
             '3m': 3,
             '5m': 5,
             '15m': 15,
             '30m': 30,
             '1h': 60,
             '2h': 120,
             '4h': 240,
             '6h': 360,
             '8h': 480,
             '12h': 720,
             '1d': 1440,
             '3d': 4320,
             '1w': 10080,
             }


# def klines(symbol, timeframe='5m', limit=1500, start=None, end=None):
#     try:
#         resp = pd.DataFrame(client.klines(symbol, timeframe, limit=limit, startTime=start, endTime=end))
#         resp = resp.iloc[:, :6]
#         resp.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
#         resp = resp.set_index('Time')
#         resp.index = pd.to_datetime(resp.index, unit='ms')
#         resp = resp.astype(float)
#         return resp
#     except ClientError as error:
#         print(
#             f"Found error. status: {error.status_code}, error code: {error.error_code}, error message: {error.error_message}")






# def klines_extended(symbol, timeframe='15m', interval_days=30):
#     ms_interval = interval_days * 24 * 3600 * 1000
#     limit = ms_interval / (intervals[timeframe] * 60 * 1000)
#     steps = limit / 1500
#     first_limit = int(steps)
#     last_step = steps - int(steps)
#     last_limit = round(1500 * last_step)
#     current_time = time() * 1000
#     p = pd.DataFrame()
#     for i in range(first_limit):
#         start = int(current_time - (ms_interval - i * 1500 * intervals[timeframe] * 60 * 1000))
#         end = start + 1500 * intervals[timeframe] * 60 * 1000
#         res = klines(symbol, timeframe = timeframe, limit=1500, start=start, end=end)
#         p = pd.concat([p, res])
#     p = pd.concat([p, klines(symbol, timeframe = timeframe, limit=last_limit)])
#     return p


def kline_extended_from_file(file_path, symbol, timeframe='15m', interval_days=30):
    """
    Fetch extended kline data from a local CSV file with Binance-style data.

    Parameters:
        file_path (str): Path to the CSV file containing historical kline data.
        symbol (str): Trading pair (e.g., 'BTCUSDT').
        timeframe (str): Kline interval (e.g., '15m', '1h', '1d').
        interval_days (int): Number of days of data to fetch.

    Returns:
        DataFrame: Reformatted kline data with required columns.
    """
    try:
       # Load the data from the CSV file
        data = pd.read_csv(file_path, header=None)

        # Assign meaningful column names
        data.columns = [
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close Time', 'Base Asset Volume', 'Number of Trades', 
            'Taker Buy Volume', 'Taker Buy Base Asset Volume', 'Ignore'
        ]

        # Convert timestamps to datetime format
        data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
        data['Close Time'] = pd.to_datetime(data['Close Time'], unit='ms')

        # Convert numeric columns to floats
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Drop rows with invalid values
        data.dropna(subset=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        # Calculate the start time based on interval_days
        end_time = data['Open Time'].max()
        start_time = end_time - timedelta(days=interval_days)

        # Filter data within the specified time range
        filtered_data = data[(data['Open Time'] >= start_time) & (data['Open Time'] <= end_time)]

        # Select and return only the relevant columns
        readable_data = filtered_data[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

        return readable_data


    except Exception as e:
        print(f"Error processing the kline file: {e}")
        return pd.DataFrame()