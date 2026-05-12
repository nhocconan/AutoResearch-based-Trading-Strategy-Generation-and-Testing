# 1d_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: Use Kaufman's Adaptive Moving Average (KAMA) on daily timeframe to capture trend direction, with weekly trend filter and volume confirmation to reduce false signals. 
# Works in both bull and bear markets by requiring alignment between daily KAMA trend and weekly trend, plus volume confirmation for entry.
# Target: 15-25 trades/year on daily timeframe.

name = "1d_KAMA_Trend_With_Volume_Confirmation"
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
    volume = prices['volume'].values

    # Get weekly data for trend filter (call once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    close_weekly = df_weekly['close'].values
    sma20_weekly = pd.Series(close_weekly).rolling(window=20, min_periods=20).mean().values
    sma20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma20_weekly)

    # KAMA on daily (ER=10, Fast=2, Slow=30)
    change = np.abs(np.diff(close, k=10))
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.abs(np.diff(close))
    volatility = np.concatenate([[np.nan], volatility])
    vol10 = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(vol10 != 0, change / vol10, 0)
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        kama_val = kama[i]
        sma20w_val = sma20_weekly_aligned[i]
        vol_avg_val = vol_avg_20[i]
        close_val = close[i]

        if np.isnan(kama_val) or np.isnan(sma20w_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + weekly uptrend + volume confirmation
            if close_val > kama_val and close_val > sma20w_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + weekly downtrend + volume confirmation
            elif close_val < kama_val and close_val < sma20w_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or weekly downtrend
            if close_val < kama_val or close_val < sma20w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or weekly uptrend
            if close_val > kama_val or close_val > sma20w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals