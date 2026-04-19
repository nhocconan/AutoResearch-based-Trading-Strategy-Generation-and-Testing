#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Chaikin Money Flow + 1d Donchian(20) Breakout
# Long when: CMF > 0.15 (bullish money flow) AND price breaks above 1d Donchian upper band
# Short when: CMF < -0.15 (bearish money flow) AND price breaks below 1d Donchian lower band
# Uses volume-weighted accumulation/distribution to confirm institutional participation
# Donchian breakout provides trend-following structure from higher timeframe
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years)

name = "6h_CMF_DonchianBreakout_1d"
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
    
    # Get daily data for Donchian breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate Chaikin Money Flow (20-period) on 6h data
    # CMF = sum((Close - Low - (High - Close)) / (High - Low) * Volume) / sum(Volume)
    # Simplified: CMF = sum(((Close - Low) - (High - Close)) / (High - Low) * Volume) / sum(Volume)
    # Which equals: sum((2*Close - High - Low) / (High - Low) * Volume) / sum(Volume)
    hl_range = high - low
    # Avoid division by zero
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mf_multiplier = ((2 * close - high - low) / hl_range) * volume
    mf_volume_sum = pd.Series(mf_multiplier).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    # Avoid division by zero
    volume_sum = np.where(volume_sum == 0, 1e-10, volume_sum)
    cmf = mf_volume_sum / volume_sum
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need CMF and Donchian data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cmf[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        cmf_val = cmf[i]
        upper_break = donch_high_aligned[i]
        lower_break = donch_low_aligned[i]
        
        if position == 0:
            # Enter long: CMF > 0.15 AND price breaks above 1d Donchian high
            if cmf_val > 0.15 and price > upper_break:
                signals[i] = 0.25
                position = 1
            # Enter short: CMF < -0.15 AND price breaks below 1d Donchian low
            elif cmf_val < -0.15 and price < lower_break:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when CMF <= 0 OR price breaks below 1d Donchian low
            if cmf_val <= 0 or price < lower_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when CMF >= 0 OR price breaks above 1d Donchian high
            if cmf_val >= 0 or price > upper_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals