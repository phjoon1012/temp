import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def fetch_live_btc_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",  # 1-minute interval
        "limit": 60,  # Last 60 minutes
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching data: {response.json().get('msg', 'Unknown error')}")
        return pd.DataFrame(columns=["Open Time", "Close"])

    data = response.json()
    df = pd.DataFrame(data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time",
        "Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
    ])
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
    df["Close"] = df["Close"].astype(float)
    return df[["Open Time", "Close"]]

def live_btc_chart(): 
    btc_data = fetch_live_btc_data()

    if btc_data.empty:
        return None  # Return None if no data is fetched

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=btc_data["Open Time"],
        y=btc_data["Close"],
        mode="lines+markers",
        name="BTC Price",
        line=dict(color="blue", width=2)
    ))

    fig.update_layout(
        title="실시간 BTC 가격",
        xaxis_title="시간",
        yaxis_title="가격 (USDT)",
        template="plotly_white",
        height=500
    )
    return fig

def fetch_current_btc_price():
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": "BTCUSDT"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return float(response.json()["price"])
    else:
        st.error(f"Error fetching BTC price: {response.json().get('msg', 'Unknown error')}")
        return None

