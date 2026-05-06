#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d Supertrend for mean reversion in range + trend following
# Uses Williams %R(14) on 12h to identify overbought/oversold conditions
# Confirms with 1d Supertrend(ATR=10, mult=3) for trend direction
# Uses 12h volume spike (>2x 20-bar average) for participation
# Williams %R effective in ranging markets, Supertrend filters for trending conditions
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: mean reversion in range, trend following in trends

name = "12h_WilliamsR_1dSupertrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 14 or len(df_1d) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 12h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_12h) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # Calculate 1d Supertrend (ATR=10, multiplier=3)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic ATR calculation for upper/lower bands
    atr_for_bands = atr_1d
    
    # Upper and lower bands
    upper_band = ((high_1d + low_1d) / 2) + (3 * atr_for_bands)
    lower_band = ((high_1d + low_1d) / 2) - (3 * atr_for_bands)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    for i in range(len(close_1d)):
        if np.isnan(atr_1d[i]):
            continue
        if i == 0:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close_1d[i] > supertrend[i-1]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            
            # Adjust bands based on direction
            if direction[i] == 1:
                if lower_band[i] < lower_band[i-1]:
                    lower_band[i] = lower_band[i-1]
            else:
                if upper_band[i] > upper_band[i-1]:
                    upper_band[i] = upper_band[i-1]
    
    # Calculate volume spike filter (>2x 20-bar average)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 12h timeframe (primary)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) AND uptrend (direction = 1) AND volume spike
            if (williams_r_aligned[i] < -80 and direction_aligned[i] == 1 and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) AND downtrend (direction = -1) AND volume spike
            elif (williams_r_aligned[i] > -20 and direction_aligned[i] == -1 and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R overbought (> -20) OR trend reversal
            if williams_r_aligned[i] > -20 or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R oversold (< -80) OR trend reversal
            if williams_r_aligned[i] < -80 or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals