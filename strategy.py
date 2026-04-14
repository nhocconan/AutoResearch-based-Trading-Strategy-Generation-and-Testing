#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R for mean reversion entries and 12h Supertrend for trend filtering.
# Williams %R identifies overbought/oversold conditions on daily timeframe.
# Supertrend from 12h ensures trades align with higher timeframe trend.
# Volume confirmation (>1.5x 20-period average) filters low-probability signals.
# Designed to work in both bull and bear markets by using 12h trend filter to avoid counter-trend trades.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Supertrend on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR calculation
    atr_period = 10
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    factor = 3.0
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    supertrend = np.zeros_like(close_12h)
    dir_ = np.ones_like(close_12h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    dir_[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upper_band[i-1]:
            dir_[i] = 1
        elif close_12h[i] < lower_band[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
            if dir_[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if dir_[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if dir_[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align 12h Supertrend to 4h timeframe
    supertrend_12h = supertrend
    dir_12h = dir_
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    dir_aligned = align_htf_to_ltf(prices, df_12h, dir_12h.astype(float))
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need Williams %R and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(dir_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for mean reversion entries based on Williams %R extremes
            # Only trade in direction of 12h Supertrend (trend filter)
            
            # Long: Williams %R oversold (< -80) AND 12h Supertrend uptrend
            if (williams_r_aligned[i] < -80 and 
                dir_aligned[i] == 1 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) AND 12h Supertrend downtrend
            elif (williams_r_aligned[i] > -20 and 
                  dir_aligned[i] == -1 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or 12h Supertrend turns down
            if (williams_r_aligned[i] > -50 or 
                dir_aligned[i] == -1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or 12h Supertrend turns up
            if (williams_r_aligned[i] < -50 or 
                dir_aligned[i] == 1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hSupertrend_1dWilliamsR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0