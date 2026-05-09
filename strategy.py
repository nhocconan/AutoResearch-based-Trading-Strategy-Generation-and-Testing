#!/usr/bin/env python3
# Hypothesis: 1h timeframe with 4h Donchian channel breakout for direction and 1h volume confirmation for entry timing.
# Uses 4h Donchian(20) for trend direction and 1h volume spike (>1.5x 20-period average) for entry confirmation.
# Enters long when price breaks above 4h Donchian high with volume confirmation, short when breaks below 4h Donchian low with volume confirmation.
# Exits when price returns to 4h Donchian midpoint or volume drops below average.
# Designed to work in both bull and bear markets by using volatility-based breakouts with volume confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) with size 0.20.

name = "1h_Donchian20_4hVolConf"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate 4h Donchian channel (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    
    # Calculate volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high with volume spike
            if close[i] > donch_high_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian low with volume spike
            elif close[i] < donch_low_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 4h Donchian midpoint or volume drops below average
            if close[i] <= donch_mid_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to 4h Donchian midpoint or volume drops below average
            if close[i] >= donch_mid_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals