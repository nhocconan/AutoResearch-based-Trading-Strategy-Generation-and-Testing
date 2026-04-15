#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w trend filter
# Designed for low trade frequency (target 15-25/year) with clear trend-following logic
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets
# Uses Donchian channels from daily, volume surge to confirm breakout strength, and weekly EMA for trend alignment
# High-probability entries in trending markets, avoids whipsaws in ranging conditions

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (primary timeframe) for Donchian calculation
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
    
    # Calculate Donchian channels (20-period) on 1d - using previous day's data to avoid look-ahead
    # Highest high of last 20 days (excluding current day)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed periods (avoid look-ahead)
    donchian_high = np.concatenate([[np.nan], high_20[:-1]])
    donchian_low = np.concatenate([[np.nan], low_20[:-1]])
    
    # Volume average (20-period on 1d) - using previous data
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg = np.concatenate([[np.nan], vol_avg[:-1]])  # shift for no look-ahead
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + uptrend + volume surge
        if (high[i] > donchian_high_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian low + downtrend + volume surge
        elif (low[i] < donchian_low_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or price returns to the middle of the channel
        elif position == 1 and close[i] < (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_20_Volume_Surge_1wTrend"
timeframe = "1d"
leverage = 1.0