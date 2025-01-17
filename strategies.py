import pandas as pd
import ta
from backtesting import Strategy

class str(Strategy):
    ema_period = 200
    rsi_period = 30

    def init(self):
        self.rsi = self.I(lambda x: ta.momentum.RSIIndicator(pd.Series(x), self.rsi_period).rsi(), self.data.Close)
        self.ema = self.I(lambda x: ta.trend.EMAIndicator(pd.Series(x), self.ema_period).ema_indicator(), self.data.Close)

    def next(self):
        price = float(self.data.Close[-1])
        if not self.position:
            if self.rsi[-2] < 30:
                self.buy(size=0.02, tp=(1 + self.tp) * price, sl=(1 - self.sl) * price)
            elif self.rsi[-2] > 70:
                self.sell(size=0.02, tp=(1 - self.tp) * price, sl=(1 + self.sl) * price)

# Strategy 2: MACD Crossover
class str2(Strategy):
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        self.macd = self.I(lambda x: ta.trend.MACD(pd.Series(x), self.fast_period, self.slow_period, self.signal_period).macd(), self.data.Close)
        self.signal = self.I(lambda x: ta.trend.MACD(pd.Series(x), self.fast_period, self.slow_period, self.signal_period).macd_signal(), self.data.Close)

    def next(self):
        price = float(self.data.Close[-1])
        if not self.position and self.macd[-1] > self.signal[-1] and self.macd[-2] <= self.signal[-2]:
            self.buy(size=0.02, tp=(1 + self.tp) * price, sl=(1 - self.sl) * price)
        elif self.position and self.macd[-1] < self.signal[-1] and self.macd[-2] >= self.signal[-2]:
            self.position.close()

# Strategy 3: Bollinger Bands Reversion
class str3(Strategy):
    bb_period = 20

    def init(self):
        self.bb_h = self.I(lambda x: ta.volatility.BollingerBands(pd.Series(x), self.bb_period).bollinger_hband(), self.data.Close)
        self.bb_l = self.I(lambda x: ta.volatility.BollingerBands(pd.Series(x), self.bb_period).bollinger_lband(), self.data.Close)

    def next(self):
        price = float(self.data.Close[-1])
        if not self.position:
            if price < self.bb_l[-1]:
                self.buy(size=0.02, tp=(1 + self.tp) * price, sl=(1 - self.sl) * price)
            elif price > self.bb_h[-1]:
                self.sell(size=0.02, tp=(1 - self.tp) * price, sl=(1 + self.sl) * price)

class IntegratedStrategy(Strategy):
    ema_period = 200
    rsi_period = 14
    fast_period = 12
    slow_period = 26
    signal_period = 9
    bb_period = 20  

    def init(self):
        # Indicators from str1 (RSI & EMA)
        self.rsi = self.I(lambda x: ta.momentum.RSIIndicator(pd.Series(x), self.rsi_period).rsi(), self.data.Close)
        self.ema = self.I(lambda x: ta.trend.EMAIndicator(pd.Series(x), self.ema_period).ema_indicator(), self.data.Close)

        # Indicators from str2 (MACD)
        self.macd = self.I(lambda x: ta.trend.MACD(pd.Series(x), self.fast_period, self.slow_period, self.signal_period).macd(), self.data.Close)
        self.signal = self.I(lambda x: ta.trend.MACD(pd.Series(x), self.fast_period, self.slow_period, self.signal_period).macd_signal(), self.data.Close)

        # Indicators from str3 (Bollinger Bands)
        self.bb_h = self.I(lambda x: ta.volatility.BollingerBands(pd.Series(x), self.bb_period).bollinger_hband(), self.data.Close)
        self.bb_l = self.I(lambda x: ta.volatility.BollingerBands(pd.Series(x), self.bb_period).bollinger_lband(), self.data.Close)

    def next(self):
        price = float(self.data.Close[-1])
        
        # Conditions for a long position (buy)
        buy_conditions = [
            self.rsi[-2] < 30,  # RSI is oversold
            price > self.ema[-1],  # Price above EMA
            self.macd[-1] > self.signal[-1] and self.macd[-2] <= self.signal[-2],  # MACD crossover
            price < self.bb_l[-1]  # Price below Bollinger Band lower limit
        ]
        
        # Count satisfied buy conditions
        satisfied_buy_conditions = sum(buy_conditions)

        # Trigger a buy if at least two conditions are satisfied
        if not self.position and satisfied_buy_conditions >= 2:
            self.buy(size=0.02, tp=(1 + self.tp) * price, sl=(1 - self.sl) * price)

        # Conditions for a short position (sell)
        sell_conditions = [
            self.rsi[-2] > 70,  # RSI is overbought
            price < self.ema[-1],  # Price below EMA
            self.macd[-1] < self.signal[-1] and self.macd[-2] >= self.signal[-2],  # MACD crossover
            price > self.bb_h[-1]  # Price above Bollinger Band upper limit
        ]
        
        # Count satisfied sell conditions
        satisfied_sell_conditions = sum(sell_conditions)

        # Trigger a sell if at least two conditions are satisfied
        if not self.position and satisfied_sell_conditions >= 2:
            self.sell(size=0.02, tp=(1 - self.tp) * price, sl=(1 + self.sl) * price)