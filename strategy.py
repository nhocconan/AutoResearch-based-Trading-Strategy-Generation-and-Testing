#!/usr/bin/env python3
# 1D_Weekly_Pivot_Trend_Volume
# Hypothesis: Use weekly Camarilla pivot levels (R1/S1) for entries on 1d chart, confirmed by weekly EMA trend and volume spikes (>2x 20-period average). Enter long at R1 break with uptrend, short at S1 break with downtrend. Exit at close crossing weekly EMA. Targets 15-25 trades/year to minimize fee drag and work in both bull/bear via trend filter.

name = "1D_Weekly_Pivot_Trend_Volume"
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

    # Get weekly data for Camarilla pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate Camarilla pivot levels (R1, S1) from previous weekly candle
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = close_1w + (high_1w - low_1w) * 1.1 / 12.0
    s1 = close_1w - (high_1w - low_1w) * 1.1 / 12.0

    # Align Camarilla levels to 1d timeframe (available after weekly candle closes)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + price > weekly EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + price < weekly EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below weekly EMA50
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above weekly EMA50
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals