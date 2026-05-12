#3/26/2025, 2:10:41 PM
#!/usr/bin/env python3

# 1d_1W_RSI_Reversion_with_WeeklyTrend
# Hypothesis: Buy when daily RSI < 30 in weekly uptrend, sell when RSI > 70 in weekly downtrend.
# RSI mean reversion works in ranging markets, weekly trend filter ensures alignment with higher timeframe momentum.
# Designed for low frequency (10-25 trades/year) to minimize fee drag.

name = "1d_1W_RSI_Reversion_with_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: RSI oversold in weekly uptrend
            if rsi[i] < 30 and bullish_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought in weekly downtrend
            elif rsi[i] > 70 and bearish_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or weekly trend turns bearish
            if rsi[i] > 70 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or weekly trend turns bullish
            if rsi[i] < 30 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals