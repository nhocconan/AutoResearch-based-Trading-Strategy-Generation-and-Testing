#!/usr/bin/env python3
"""
6h_supertrend_volume_12h1d_v1
Hypothesis: Use Supertrend(10,3) on 12h for trend direction, Supertrend(10,3) on 1d for regime filter, and volume confirmation on 6h. Enter long when both timeframes are bullish and volume confirms; short when both are bearish and volume confirms. Exit when either timeframe turns opposite or volume drops. Designed for 6-12 trades per month (72-144/year) to minimize fee drift while capturing sustained trends. Works in bull (rides uptrends) and bear (rides downtrends) by requiring alignment across timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_supertrend_volume_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator."""
    if len(high) < period:
        return np.full(len(close), np.nan), np.full(len(close), np.nan)
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    
    # Calculate upper and lower bands
    hl2 = (high + low) / 2
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(len(close), np.nan)
    direction = np.full(len(close), np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Set first value
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
            direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on 12h
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    st_12h, dir_12h = supertrend(h_12h, l_12h, c_12h, 10, 3)
    st_12h_aligned = align_htf_to_ltf(prices, df_12h, st_12h)
    dir_12h_aligned = align_htf_to_ltf(prices, df_12h, dir_12h)
    
    # Calculate Supertrend on 1d
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    st_1d, dir_1d = supertrend(h_1d, l_1d, c_1d, 10, 3)
    st_1d_aligned = align_htf_to_ltf(prices, df_1d, st_1d)
    dir_1d_aligned = align_htf_to_ltf(prices, df_1d, dir_1d)
    
    # Volume filter: 6h volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # Fill NaN with 1.0 (no volume filter)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Check if Supertrend values are available
        if np.isnan(dir_12h_aligned[i]) or np.isnan(dir_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend alignment
        bullish_aligned = (dir_12h_aligned[i] == 1) and (dir_1d_aligned[i] == 1)
        bearish_aligned = (dir_12h_aligned[i] == -1) and (dir_1d_aligned[i] == -1)
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.3
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when either timeframe turns bearish
            if not bullish_aligned:
                exit_long = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.8:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when either timeframe turns bullish
            if not bearish_aligned:
                exit_short = True
            # Exit when volume drops significantly
            elif vol_ratio[i] < 0.8:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry when both timeframes bullish and volume confirms
            if bullish_aligned and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short entry when both timeframes bearish and volume confirms
            elif bearish_aligned and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals