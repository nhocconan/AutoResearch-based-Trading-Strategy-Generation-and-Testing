#!/usr/bin/env python3
"""
6h_12h_1d_Pivot_Fade_v1
Hypothesis: On 6h timeframe, fade price moves that reach daily pivot extremes (R4/S4) with volume confirmation, 
using 12h trend filter to avoid counter-trend trades in strong trends. Works in both bull and bear markets 
by fading extremes in ranging conditions while respecting higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Pivot_Fade_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot levels from previous day
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Calculate support/resistance levels
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    r4 = prev_high + 3 * (prev_high - prev_low)
    s4 = prev_low - 3 * (prev_high - prev_low)
    
    # Create arrays for each level and align to 6h timeframe
    def create_aligned_array(value):
        arr = np.full(len(df_1d), value)
        return align_htf_to_ltf(prices, df_1d, arr)
    
    r1_aligned = create_aligned_array(r1)
    s1_aligned = create_aligned_array(s1)
    r2_aligned = create_aligned_array(r2)
    s2_aligned = create_aligned_array(s2)
    r3_aligned = create_aligned_array(r3)
    s3_aligned = create_aligned_array(s3)
    r4_aligned = create_aligned_array(r4)
    s4_aligned = create_aligned_array(s4)
    
    # Volume filter: current volume > 1.5x 30-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=30, min_periods=30).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade counter to 12h trend when at extremes
        # In uptrend (price > EMA50), only allow shorts at resistance
        # In downtrend (price < EMA50), only allow longs at support
        is_uptrend = close[i] > ema_50_12h_aligned[i]
        is_downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Fade at S4/R4 with volume confirmation
        long_setup = (low[i] <= s4_aligned[i] and vol_ratio[i] > 1.5 and is_downtrend)
        short_setup = (high[i] >= r4_aligned[i] and vol_ratio[i] > 1.5 and is_uptrend)
        
        # Exit at midpoint (S1/R1) or opposite extreme
        long_exit = (close[i] >= (s1_aligned[i] + r1_aligned[i]) / 2) or (high[i] >= r4_aligned[i])
        short_exit = (close[i] <= (r1_aligned[i] + s1_aligned[i]) / 2) or (low[i] <= s4_aligned[i])
        
        # Signal logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals