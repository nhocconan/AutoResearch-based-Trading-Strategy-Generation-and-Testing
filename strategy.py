#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1d volume confirmation and 1w trend filter
# Designed for low trade frequency (target 10-25/year) with clear trend-following logic
# Works in both bull (breakout above upper band in uptrend) and bear (breakdown below lower band in downtrend) markets
# Uses Donchian channels from daily, volume spike to confirm breakout strength, and weekly EMA for trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) from previous day to avoid look-ahead
    # Using previous day's data to calculate bands for today's breakout
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Calculate 20-period Donchian high and low from previous data
    donchian_high = pd.Series(prev_high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe (since we're using daily data, alignment is straightforward)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size as fraction of capital
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + uptrend + volume spike
        if (high[i] > donchian_high_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian low + downtrend + volume spike
        elif (low[i] < donchian_low_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or price returns to the middle of the channel
        elif position == 1 and close[i] <= (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_1dVolume_1wEMA_Trend"
timeframe = "1d"
leverage = 1.0