#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_camarilla_pivot_reversal_v1
# Fade at Camarilla R3/S3 levels, breakout continuation at R4/S4 on 6h timeframe.
# Uses 1d Camarilla levels for structure and 6h volume confirmation.
# Works in both bull and bear markets: mean reversion at extremes, trend following on breaks.
# Low trade frequency expected (15-25/year) due to strict level + volume filter.
name = "6h_1d_camarilla_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    rng = high_1d - low_1d
    r4 = close_1d + 1.5 * rng
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align levels to 6h timeframe (previous day's levels are known at open)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any level not ready
        if np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 20-period average
        vol_filter = volume[i] > vol_ma[i]
        
        # Fade at S3/R3 (mean reversion)
        long_fade = (low[i] <= s3_6h[i] and close[i] > s3_6h[i]) and vol_filter
        short_fade = (high[i] >= r3_6h[i] and close[i] < r3_6h[i]) and vol_filter
        
        # Breakout continuation at S4/R4 (trend following)
        long_break = (high[i] > s4_6h[i] and close[i] > s4_6h[i]) and vol_filter
        short_break = (low[i] < r4_6h[i] and close[i] < r4_6h[i]) and vol_filter
        
        # Exit conditions
        exit_long = (high[i] >= r3_6h[i] and close[i] < r3_6h[i]) or (low[i] <= s3_6h[i] and close[i] > s3_6h[i])
        exit_short = (low[i] <= s3_6h[i] and close[i] > s3_6h[i]) or (high[i] >= r3_6h[i] and close[i] < r3_6h[i])
        
        if (long_fade or long_break) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_fade or short_break) and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals