#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 breakouts on 12h with daily trend filter and volume confirmation to capture intermediate-term trends. Works in both bull and bear markets by filtering trades with higher timeframe trend and requiring volume confirmation, reducing false signals during choppy periods. Target: 12-37 trades/year.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Get daily data for Camarilla calculation and trend filter (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (high_prev + low_prev + close_prev) / 3
    range_val = high_prev - low_prev
    
    # Calculate Camarilla levels (R1, S1)
    r1 = close_prev + (range_val * 1.1 / 12)
    s1 = close_prev - (range_val * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34_val = ema34_daily_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + daily uptrend + volume confirmation
            if close[i] > r1_val and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + daily downtrend + volume confirmation
            elif close[i] < s1_val and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or close below daily EMA34
            if close[i] < s1_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or close above daily EMA34
            if close[i] > r1_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals