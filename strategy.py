#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_12hTrend_Volume
# Hypothesis: Use 4h Camarilla R1/S1 breakouts with 12h EMA trend filter and volume confirmation.
# Camarilla levels provide intraday support/resistance; EMA filter ensures alignment with higher-timeframe trend.
# Works in bull (follows breaks with bullish 12h trend) and bear (avoids bullish breaks in bearish 12h trend).
# Target: 100-200 total trades over 4 years = 25-50/year.

name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume"
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

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Calculate Camarilla levels on 4h data
    # Pivot = (H+L+C)/3
    pivot = (high + low + close) / 3
    range_hl = high - low
    # R1 = C + (H-L)*1.1/12
    r1 = close + range_hl * 1.1 / 12
    # S1 = C - (H-L)*1.1/12
    s1 = close - range_hl * 1.1 / 12

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + price above 12h EMA (bullish trend) + volume spike
            if (close[i] > r1[i] and 
                close[i] > ema_50_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + price below 12h EMA (bearish trend) + volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema_50_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or price below 12h EMA
            if (close[i] < s1[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or price above 12h EMA
            if (close[i] > r1[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals