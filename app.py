import streamlit as st
from backtesting import Backtest
from helper import kline_extended_from_file
from strategies import str, str2, str3, IntegratedStrategy
from visualization import (
    plot_equity_curve,
    plot_trades_chart,
    plot_returns_bar_chart,
    plot_comparison_chart
)
# from liveBTC import live_btc_chart, fetch_current_btc_price
import time


st.set_page_config(layout="wide")

# if "show_live_chart" not in st.session_state:
#     st.session_state.show_live_chart = True  # Show live BTC chart by default

st.markdown("<h1 style='text-align: center;'>암호화폐 거래 전략 벡테스팅</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("백테스팅 변수")
    symbol = st.selectbox(
        "거래 화폐:",
        options=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"],
        index=0
    )
    timeframe = st.selectbox("시간(봉):", ["1m", "5m", "15m", "1h", "4h", "1d"])
    interval = st.slider("백테스팅 범위 (일):", 1, 365, 30)
    cash = st.number_input("초기자금 (USD):", value=1000000, step=10000)
    margin = st.slider("마진:", 1, 50, 10)
    commission = st.slider("수수료 (%):", 0.01, 1.0, 0.07) / 100
    tp = st.slider("이익 실현 (%):", 0.01, 10.0, 3.0) / 100
    sl = st.slider("손실 제한 (%):", 0.01, 10.0, 2.0) / 100

    st.header("전략 선택")
    strategy_options = {
        "RSI & 지수이동평균": str,
        "MACD 크로스오버 Strategy": str2,
        "볼린저 밴드": str3,
        "통합 전략": IntegratedStrategy
    }
    strategy_name = st.selectbox("전략:", list(strategy_options.keys()))
    selected_strategy = strategy_options[strategy_name]
    run_backtest = st.button("백테스팅 실행")
price_placeholder = st.empty()


# if st.session_state.show_live_chart:
#     chart = live_btc_chart()
#     if chart:
#         st.plotly_chart(chart, use_container_width=True)



if run_backtest:
    loading_message = st.empty()
    loading_message.write(f"Fetching data for {symbol} with {timeframe} timeframe over the last {interval} days...")
    
    kl = kline_extended_from_file('cleaned_adjusted_data.csv',symbol, timeframe, interval)
    selected_strategy.tp = tp
    selected_strategy.sl = sl

    bt = Backtest(kl, selected_strategy, cash=cash, margin=1 / margin, commission=commission / 100)
    stats = bt.run()
    net_profit = stats['Equity Final [$]'] - cash

    trades = stats['_trades']
    if not trades.empty:
        win_trades = trades[trades['PnL'] > 0]
        win_rate = (len(win_trades) / len(trades)) * 100
        num_trades = len(trades)
    else:
        win_rate = 0.0
        num_trades = 0

    loading_message.empty()

    st.subheader("백테스팅 요약")
    col1, col3, col4, col5, col6 = st.columns(5)
    col1.metric("원금 대비 총 이익 [%]", f"{stats['Return [%]']:.2f} %")
    col3.metric("최대 손실 [%]", f"{stats['Max. Drawdown [%]']:.2f} %")
    col4.metric("승률 [%]", f"{win_rate:.2f} %")
    col5.metric("거래 횟수 [회]", f"{num_trades} 회")
    col6.metric("순이익 [$]", f"{net_profit:.2f} $")

    plot_trades_chart(kl, trades)
    plot_equity_curve(stats['_equity_curve'])
    plot_returns_bar_chart(trades)

    start_price = kl['Close'].iloc[0]
    historical_data = kl[['Close']].copy()
    historical_data['Open Time'] = kl.index

    plot_comparison_chart(stats['_equity_curve'], historical_data)

    # col_left, col_right = st.columns(2)
    # with col_left:
    #     st.markdown("### 주요 항목")
    #     st.dataframe(stats.drop(['_equity_curve', '_trades', '_strategy'], axis=0, errors='ignore'))
    # with col_right:
    #     st.subheader("거래 목록")
    #     if trades.empty:
    #         st.info("No trades were executed during the backtest.")
    #     else:
    #         st.dataframe(trades)
    st.subheader("거래 목록")
    if trades.empty:
        st.info("No trades were executed during the backtest.")
    else:
        st.dataframe(trades)


    # Load and display the plot
    plot_file = "LSTM_v2.png"
    st.subheader("BTC 가격 예측 (LSTM 모델)")
    st.image(plot_file, caption="BTC 가격 예측")
