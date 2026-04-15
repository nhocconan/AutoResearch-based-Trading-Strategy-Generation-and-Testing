#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(weekly_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = weekly_volume / (vol_ma_20 + 1e-10)
    
    # Align HTF indicators to 1d timeframe with proper delay
    highest_20_1d = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_1d = align_htf_to_ltf(prices, df_1w, lowest_20)
    volume_ratio_1d = align_htf_to_ltf(prices, df_1w, volume_ratio)
    
    signals = np.zeros(n)
    
    # Weekly Donchian breakout with volume confirmation
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_1d[i]) or np.isnan(lowest_20_1d[i]) or 
            np.isnan(volume_ratio_1d[i])):
            signals[i] = 0.0
            continue
        
        # Long: break above weekly Donchian high with volume
        if (close[i] > highest_20_1d[i] and 
            volume_ratio_1d[i] > 1.5):
            signals[i] = 0.25
            
        # Short: break below weekly Donchian low with volume
        elif (close[i] < lowest_20_1d[i] and 
              volume_ratio_1d[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0