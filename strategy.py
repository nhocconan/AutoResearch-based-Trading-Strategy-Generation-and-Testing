#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Strict
Camarilla pivot breakout with volume spike and volume regime filter:
- Long when close breaks above R1 + volume spike above 20-period average
- Short when close breaks below S1 + volume spike above 20-period average
- Only trade in low-volume regime (volume < 30-period median) to avoid chop
- Uses 1d Camarilla levels for structure
- Designed for 20-40 trades/year per symbol
Works in bull (breakouts) and bear (breakdowns) markets
"""

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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First day will have NaN due to roll, handled by isnan check later
    
    r1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    s1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume indicators
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_median_30 = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need volume median
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(vol_median_30[i])):
            signals[i] = 0.0
            continue
        
        # Volume conditions
        volume_spike = volume[i] > vol_ma_20[i] * 2.0  # volume spike > 2x average
        low_volume_regime = volume[i] < vol_median_30[i]  # avoid high volume chop
        
        if position == 0:
            # Long: close breaks above R1 + volume spike + low volume regime
            if close[i] > r1_4h[i] and volume_spike and low_volume_regime:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S1 + volume spike + low volume regime
            elif close[i] < s1_4h[i] and volume_spike and low_volume_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close breaks below S1 (reversal signal)
            if close[i] < s1_4h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close breaks above R1 (reversal signal)
            if close[i] > r1_4h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Strict"
timeframe = "4h"
leverage = 1.0