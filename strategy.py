# 4h_SupportResistance_Fibonacci_Extension
# Hypothesis: Use 1D swing high/low to calculate Fibonacci extension levels (127.2%, 161.8%, 261.8%) as dynamic support/resistance.
# Enter long when price breaks above 127.2% extension of prior swing low with bullish 1D trend and volume spike.
# Enter short when price breaks below 61.8% retracement of prior swing high with bearish 1D trend and volume spike.
# Exit when price crosses back below/above the swing point level.
# This captures momentum after pullbacks in trending markets while avoiding false breakouts.

name = "4h_SupportResistance_Fibonacci_Extension"
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

    # Get 1d data for swing points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate swing high/low on 1D (pivot points)
    swing_high = np.full_like(high_1d, np.nan)
    swing_low = np.full_like(low_1d, np.nan)
    
    for i in range(2, len(high_1d)-2):
        # Swing high: higher than 2 bars on each side
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = high_1d[i]
        # Swing low: lower than 2 bars on each side
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = low_1d[i]

    # Forward fill swing points to use most recent
    for i in range(1, len(swing_high)):
        if np.isnan(swing_high[i]):
            swing_high[i] = swing_high[i-1]
        if np.isnan(swing_low[i]):
            swing_low[i] = swing_low[i-1]

    # Calculate Fibonacci levels
    range_1d = swing_high - swing_low
    fib_ext_127 = swing_low + range_1d * 1.272  # 127.2% extension
    fib_ret_618 = swing_high - range_1d * 0.618  # 61.8% retracement

    # Align Fibonacci levels to 4h timeframe
    fib_ext_127_aligned = align_htf_to_ltf(prices, df_1d, fib_ext_127)
    fib_ret_618_aligned = align_htf_to_ltf(prices, df_1d, fib_ret_618)
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low)

    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(fib_ext_127_aligned[i]) or np.isnan(fib_ret_618_aligned[i]) or 
            np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above 127.2% extension + price > 1D EMA34 + volume spike
            if (close[i] > fib_ext_127_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below 61.8% retracement + price < 1D EMA34 + volume spike
            elif (close[i] < fib_ret_618_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below swing low (invalidates uptrend)
            if close[i] < swing_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above swing high (invalidates downtrend)
            if close[i] > swing_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals