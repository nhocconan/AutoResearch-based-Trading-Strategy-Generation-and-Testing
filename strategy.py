#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend
# Hypothesis: Use Camarilla pivot levels from daily data combined with 1-week EMA trend filter.
# Long when price breaks above daily R1 with volume spike and 1w EMA20 uptrend.
# Short when price breaks below daily S1 with volume spike and 1w EMA20 downtrend.
# Exit on mean reversion to daily pivot point (PP). Designed for low turnover (10-20 trades/year)
# to avoid fee drag and work in both bull and bear markets via trend filter.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend"
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
    volume = prices['volume'].values

    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)

    # Calculate Camarilla pivot levels for each day using prior day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are close, high, low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_pp = prev_close  # Not used directly but for reference
    camarilla_r1 = prev_close + range_val * 1.1 / 12
    camarilla_s1 = prev_close - range_val * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe (already aligned via shift)
    # Since we used shift(1), values are already for current day
    r1 = camarilla_r1
    s1 = camarilla_s1
    pp = camarilla_pp  # daily pivot point

    # Get 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Volume confirmation: current volume > 2.0 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(pp[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above daily R1 with volume spike and 1w EMA20 uptrend
            if close[i] > r1[i] and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below daily S1 with volume spike and 1w EMA20 downtrend
            elif close[i] < s1[i] and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below daily pivot (mean reversion to center)
            if close[i] < pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above daily pivot
            if close[i] > pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals