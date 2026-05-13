#!/usr/bin/env python3
# 1d_KAMA_WeeklyTrend_Volume
# Hypothesis: Use Kaufman's Adaptive Moving Average (KAMA) on daily timeframe to determine trend, with weekly trend filter and volume confirmation for entries.
# Long when KAMA turns upward (bullish) with weekly trend confirmation and volume spike, short when KAMA turns downward (bearish) with weekly trend confirmation and volume spike.
# Exit when KAMA reverses direction.
# Combines adaptive trend following with weekly confirmation to reduce whipsaws in both bull and bear markets.

name = "1d_KAMA_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Calculate KAMA on daily close
    # KAMA parameters: ER (Efficiency Ratio) period=10, Fast=2, Slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(1, len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smooth ER over 10 periods
    er_smooth = pd.Series(er).ewm(alpha=2/(10+1), adjust=False).fillna(0).values
    # Smoothing constants
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    sc = (er_smooth * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after KAMA warmup
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up (current > previous) + price above weekly EMA50 (uptrend) + volume spike
            if (kama[i] > kama[i-1] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down (current < previous) + price below weekly EMA50 (downtrend) + volume spike
            elif (kama[i] < kama[i-1] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals