#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses 6h as primary timeframe with weekly pivot bias and volume confirmation
# Long when: price breaks above Donchian upper band AND weekly pivot > previous close AND volume spike
# Short when: price breaks below Donchian lower band AND weekly pivot < previous close AND volume spike
# Donchian(20): 20-period high/low channel
# Weekly pivot: (weekly high + weekly low + weekly close) / 3
# Volume confirmation: current volume > 2.0x 20-period average
# Target: 15-35 trades/year per symbol (~60-140 total over 4 years)

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate Donchian channels on 6h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll.iloc[i]) or np.isnan(low_roll.iloc[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = high_roll.iloc[i]
        lower_band = low_roll.iloc[i]
        pivot = pivot_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper band AND pivot > previous close AND volume spike
            if price > upper_band and pivot > close[i-1] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band AND pivot < previous close AND volume spike
            elif price < lower_band and pivot < close[i-1] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower band
            if price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above upper band
            if price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals