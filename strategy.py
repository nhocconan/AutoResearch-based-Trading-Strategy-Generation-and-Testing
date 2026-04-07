#!/usr/bin/env python3
"""
12h Camarilla Pivot with 1d Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels act as strong support/resistance.
Using 1d EMA100 as trend filter ensures trades follow the daily trend.
Volume spikes confirm institutional participation at key levels.
Designed for 12h timeframe to target 50-150 total trades over 4 years.
Works in both bull and bear markets by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume Spike Detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # 1d EMA100 Trend Filter
    df_1d = get_htf_data(prices, '1d')
    ema_100 = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(ema_100_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 12h bar
        # Using previous day's OHLC (from 1d data)
        if i < 24:  # Need at least 2 days of 12h data to get previous day
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC from 1d data
        # 12h bars per day = 2
        prev_day_idx = (i // 2) - 1  # Previous day index in 1d data
        if prev_day_idx < 0 or prev_day_idx >= len(df_1d):
            signals[i] = 0.0
            continue
            
        prev_high = df_1d['high'].iloc[prev_day_idx]
        prev_low = df_1d['low'].iloc[prev_day_idx]
        prev_close = df_1d['close'].iloc[prev_day_idx]
        
        # Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Resistance levels
        r3 = prev_close + (range_val * 1.1 / 2)
        r4 = prev_close + (range_val * 1.1)
        # Support levels
        s3 = prev_close - (range_val * 1.1 / 2)
        s4 = prev_close - (range_val * 1.1)
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA100
            if close[i] < ema_100_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA100
            if close[i] > ema_100_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price rebounds from S3/S4 with volume spike and above EMA100
            if ((close[i] >= s3 or close[i] >= s4) and 
                close[i] > ema_100_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price rejected at R3/R4 with volume spike and below EMA100
            elif ((close[i] <= r3 or close[i] <= r4) and 
                  close[i] < ema_100_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals