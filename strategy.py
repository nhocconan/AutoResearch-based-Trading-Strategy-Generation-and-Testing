# 4h_Keltner_Breakout_Volume_Trend
# Hypothesis: Use Keltner Channel breakout (20,2.0) with 1d EMA trend filter and volume spike.
# Enter long when price breaks above upper band with 1d EMA uptrend and volume > 1.5x average.
# Enter short when price breaks below lower band with 1d EMA downtrend and volume spike.
# Exit when price closes back inside the Keltner Channel.
# This combines volatility-based breakout with trend and volume filters to reduce false signals.
# Target: 20-30 trades/year on 4h to minimize fee drag while capturing strong trends.

name = "4h_Keltner_Breakout_Volume_Trend"
timeframe = "4h"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Keltner Channel (20, 2.0) on 4h data
    # ATR(20)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # EMA(20) for middle band
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Upper and lower bands
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above upper band + price > 1d EMA34 + volume spike
            if (close[i] > upper[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below lower band + price < 1d EMA34 + volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back inside Keltner Channel (below upper band)
            if close[i] < upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back inside Keltner Channel (above lower band)
            if close[i] > lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals