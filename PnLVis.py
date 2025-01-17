import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from datetime import datetime

# Define column names
columns = [
    "Open Time", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Base Asset Volume", "Number of Trades",
    "Taker Buy Volume", "Taker Buy Base Asset Volume", "Ignore"
]

# Load the dataset
file_path = "/content/BTCUSDT-1h-2024-12.csv"  # Replace with your CSV file path
btc_data = pd.read_csv(file_path, names=columns, header=None)

# Convert timestamps to datetime format
btc_data["Open Time"] = pd.to_datetime(btc_data["Open Time"], unit='ms')
btc_data["Close"] = btc_data["Close"].astype(float)

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("BTC Profit Percentage Visualization", style={'textAlign': 'center'}),
    dcc.Graph(id="profit-graph"),
    dcc.DatePickerRange(
        id="date-picker-range",
        start_date=btc_data["Open Time"].dt.date.min(),
        end_date=btc_data["Open Time"].dt.date.max(),
        display_format="YYYY-MM-DD",
        style={"margin": "20px"}
    ),
    html.Div(id="output-data", style={"textAlign": "center", "marginTop": "20px"})
])

# Callback to update graph and profit calculations based on date range
@app.callback(
    [Output("profit-graph", "figure"),
     Output("output-data", "children")],
    [Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date")]
)
def update_graph(start_date, end_date):
    # Filter data based on selected date range
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    filtered_data = btc_data[(btc_data["Open Time"] >= start_date) & (btc_data["Open Time"] <= end_date)]

    # If the range is empty, return a placeholder
    if filtered_data.empty:
        return {}, "No data available for the selected date range."

    # Calculate profit percentage relative to the first day's close price
    bought_price = filtered_data.iloc[0]["Close"]
    filtered_data["Profit %"] = ((filtered_data["Close"] - bought_price) / bought_price) * 100

    # Calculate profit percentage for the last day in the range
    last_day_price = filtered_data.iloc[-1]["Close"]
    profit_percentage = ((last_day_price - bought_price) / bought_price) * 100

    # Create the graph
    fig = px.line(filtered_data, x="Open Time", y="Profit %", title="Profit Percentage Over Time")
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")

    # Return the graph and summary text
    summary = f"Profit percentage from {start_date.date()} to {end_date.date()} is {profit_percentage:.2f}%"
    return fig, summary

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)