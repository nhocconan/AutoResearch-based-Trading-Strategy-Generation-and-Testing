#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian breakout with volume confirmation and weekly EMA filter
# Uses weekly Donchian channel breakouts on daily chart for trend capture,
# volume to confirm breakout strength, and weekly EMA20 for trend filter.
# Designed for 1d timeframe to capture major moves with low trade frequency.
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (primary timeframe) for price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for weekly EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA20 on weekly for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + price above weekly EMA20
        if (close[i] > donch_high_1d_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema20_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + price below weekly EMA20
        elif (close[i] < donch_low_1d_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema20_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and close[i] < donch_low_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Volume_EMA_Filter"
timeframe = "1d"
leverage = 1.0