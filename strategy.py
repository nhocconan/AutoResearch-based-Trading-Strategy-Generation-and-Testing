#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot level touches with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (derived from 1d OHLC) act as strong support/resistance.
# Price touching these levels with volume confirmation indicates potential reversals.
# In bull markets, buy at support levels (S1, S2) with volume. In bear markets, sell at resistance levels (R1, R2) with volume.
# Uses 1d EMA50 for trend filter to align with higher timeframe direction. Target: ~20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # These levels are based on the previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d_vals) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # Resistance levels
    r1 = pivot + (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    # Support levels
    s1 = pivot - (range_1d * 1.1 / 12)
    s2 = pivot - (range_1d * 1.1 / 6)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Align all levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Camarilla level touch + volume + trend alignment
        # Long when price touches support levels in uptrend
        if ((close[i] <= s1_aligned[i] * 1.001 and close[i] >= s1_aligned[i] * 0.999) or
            (close[i] <= s2_aligned[i] * 1.001 and close[i] >= s2_aligned[i] * 0.999)) and \
           vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        # Short when price touches resistance levels in downtrend
        elif ((close[i] >= r1_aligned[i] * 0.999 and close[i] <= r1_aligned[i] * 1.001) or
              (close[i] >= r2_aligned[i] * 0.999 and close[i] <= r2_aligned[i] * 1.001)) and \
             vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend change or opposite level touch
        elif position == 1 and (not uptrend or 
                               (close[i] >= r1_aligned[i] * 0.999 and close[i] <= r1_aligned[i] * 1.001)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or 
                                (close[i] <= s1_aligned[i] * 1.001 and close[i] >= s1_aligned[i] * 0.999)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals