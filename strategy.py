#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d volume confirmation and 1w EMA trend filter
# Designed to capture trends with clear entry/exit rules, using Alligator jaws/teeth/lips for trend direction
# Works in both bull (buy when green alignment above red) and bear (sell when red alignment above green) markets
# Uses Williams Alligator (13,8,5 SMAs) from 6h, volume spike from 1d to confirm interest, and weekly EMA for trend alignment
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 6h: Jaw (13), Teeth (8), Lips (5) - all SMAs
    jaw_6h = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    teeth_6h = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    lips_6h = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_6h_aligned[i]) or np.isnan(teeth_6h_aligned[i]) or 
            np.isnan(lips_6h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Green alignment (bullish): Lips > Teeth > Jaw
        green_alignment = (lips_6h_aligned[i] > teeth_6h_aligned[i] > jaw_6h_aligned[i])
        # Red alignment (bearish): Jaw > Teeth > Lips
        red_alignment = (jaw_6h_aligned[i] > teeth_6h_aligned[i] > lips_6h_aligned[i])
        
        # Long entry: green alignment + price above lips + volume spike + uptrend filter
        if (green_alignment and 
            close[i] > lips_6h_aligned[i] and 
            volume[i] > 1.8 * vol_avg_1d_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: red alignment + price below jaws + volume spike + downtrend filter
        elif (red_alignment and 
              close[i] < jaw_6h_aligned[i] and 
              volume[i] > 1.8 * vol_avg_1d_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite alignment or price crosses midline (teeth)
        elif position == 1 and (not green_alignment or close[i] < teeth_6h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not red_alignment or close[i] > teeth_6h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dVolume_1wEMA_Trend"
timeframe = "6h"
leverage = 1.0