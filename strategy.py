#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for 20-week Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period weekly Donchian channels
    upper_20w = np.full_like(high_1w, np.nan)
    lower_20w = np.full_like(low_1w, np.nan)
    
    for i in range(len(high_1w)):
        if i < 19:
            upper_20w[i] = np.nan
            lower_20w[i] = np.nan
        else:
            upper_20w[i] = np.max(high_1w[i-19:i+1])
            lower_20w[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian to daily (wait for weekly close)
    upper_20w_aligned = align_htf_to_ltf(prices, df_1w, upper_20w)
    lower_20w_aligned = align_htf_to_ltf(prices, df_1w, lower_20w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_series = pd.Series(volume_1d)
    avg_vol_20 = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # 20 for Donchian and volume
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20w_aligned[i]) or np.isnan(lower_20w_aligned[i]) or
            np.isnan(avg_vol_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume_1d[i]  # Use daily volume
        
        if position == 0:
            # Long: price breaks above 20-week high with volume confirmation
            if price > upper_20w_aligned[i] and vol > 1.3 * avg_vol_20[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below 20-week low with volume confirmation
            elif price < lower_20w_aligned[i] and vol > 1.3 * avg_vol_20[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 20-week low
            if price < lower_20w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 20-week high
            if price > upper_20w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0