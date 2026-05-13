# 1h_Combined_Momentum_Trend
# Hypothesis: Combining 4h EMA trend direction with 1h momentum (RSI) and volume confirmation reduces false signals.
# In bull markets: 4h EMA up + RSI > 50 + volume > 1.5x average → long
# In bear markets: 4h EMA down + RSI < 50 + volume > 1.5x average → short
# Uses 4h EMA for trend direction (trades with trend) and 1h RSI for momentum timing.
# Volume filter ensures momentum is supported by participation.
# Target: 15-35 trades/year per symbol to minimize fee drag.

name = "1h_Combined_Momentum_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # 4h EMA for trend direction
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # 1h RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h EMA uptrend + RSI > 50 + volume spike
            if (close[i] > ema_4h_aligned[i] and 
                rsi[i] > 50 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h EMA downtrend + RSI < 50 + volume spike
            elif (close[i] < ema_4h_aligned[i] and 
                  rsi[i] < 50 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h EMA downtrend or RSI < 40
            if close[i] < ema_4h_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h EMA uptrend or RSI > 60
            if close[i] > ema_4h_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals