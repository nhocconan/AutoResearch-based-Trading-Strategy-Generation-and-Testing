#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Buy when price breaks above Camarilla R1 level with daily uptrend filter (price > daily EMA34) and volume confirmation; sell when price breaks below S1 level with daily downtrend filter and volume confirmation. Exit on opposite Camarilla level touch. Camarilla levels from prior day provide robust support/resistance that works in both trending and ranging markets. Daily trend filter avoids counter-trend trades. Volume confirmation ensures momentum. Target: 25-40 trades/year.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get daily data for Camarilla levels and trend filter (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H = high, L = low, C = close of previous day
    H = df_daily['high'].values
    L = df_daily['low'].values
    C = df_daily['close'].values
    
    # Avoid division by zero when H == L
    diff = H - L
    diff_safe = np.where(diff == 0, 1e-10, diff)
    
    # Camarilla R1 = C + (H-L) * 1.1/12
    # Camarilla S1 = C - (H-L) * 1.1/12
    r1 = C + diff_safe * 1.1 / 12
    s1 = C - diff_safe * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(C).ewm(span=34, adjust=False, min_periods=34).mean().values
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
            # LONG: price breaks above R1 + daily uptrend + volume confirmation
            if close[i] > r1_val and close[i-1] <= r1_val and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + daily downtrend + volume confirmation
            elif close[i] < s1_val and close[i-1] >= s1_val and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches S1 level (opposite Camarilla level)
            if close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches R1 level (opposite Camarilla level)
            if close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals