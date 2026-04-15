#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + 1w trend filter
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and strength,
# volume spikes to confirm momentum, and weekly EMA200 to filter for primary trend.
# Works in bull markets (buy when green aligned above red) and bear markets (sell when red aligned above green).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 12h (13,8,5 SMAs shifted by 8,5,3)
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # Jaw: 13-period SMA shifted 8 bars
    
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # Teeth: 8-period SMA shifted 5 bars
    
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # Lips: 5-period SMA shifted 3 bars
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA200 on 1w for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_vals)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema200_1w_aligned[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw (green above red)
        bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish alignment: Jaw > Teeth > Lips (red above green)
        bearish = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Long entry: Bullish Alligator + volume spike + price above weekly EMA200
        if (bullish and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema200_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bearish Alligator + volume spike + price below weekly EMA200
        elif (bearish and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema200_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite Alligator alignment or loss of weekly trend filter
        elif position == 1 and (bearish or close[i] < ema200_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish or close[i] > ema200_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0