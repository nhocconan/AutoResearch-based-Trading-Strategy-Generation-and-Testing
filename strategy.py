#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with volume confirmation and 1w EMA trend filter
# Designed for low trade frequency (target 20-40/year) with clear breakout logic
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band)
# Uses 1-day Donchian channels, volume spike confirmation, and weekly EMA for trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (primary timeframe) for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) from previous day
    # Using previous day's data to avoid look-ahead
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Previous day's Donchian levels (to avoid look-ahead)
    prev_high_20 = np.concatenate([[np.nan], high_20[:-1]])
    prev_low_20 = np.concatenate([[np.nan], low_20[:-1]])
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, prev_high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, prev_low_20)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long breakout: price breaks above upper Donchian + uptrend + volume spike
        if (high[i] > high_20_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short breakdown: price breaks below lower Donchian + downtrend + volume spike
        elif (low[i] < low_20_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to opposite band
        elif position == 1 and low[i] < low_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > high_20_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0