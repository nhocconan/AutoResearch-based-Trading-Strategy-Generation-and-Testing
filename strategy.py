# 145899
#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 12h/1d timeframe filters. Uses 12h Donchian channel breakout with 1d volume confirmation.
# Long when price breaks above 12h Donchian high (20-period) with 1d volume > 1.5x 20-period average.
# Short when price breaks below 12h Donchian low (20-period) with 1d volume > 1.5x 20-period average.
# Exit when price returns to the 12h Donchian midpoint (mean of high/low channel).
# Designed to capture momentum bursts with volume confirmation while avoiding false breakouts.
# Targets 60-120 total trades over 4 years (15-30/year) with position size 0.25.
# Works in both bull and bear markets by filtering breakouts with volume (avoids low-conviction moves).

name = "6h_Donchian_Breakout_12hHighLow_1dVolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channel (20-period high/low)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian high and low
    donch_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align 12h Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # Calculate 1d volume confirmation: current volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    vol_confirm = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian high with volume confirmation
            if close[i] > donch_high_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian low with volume confirmation
            elif close[i] < donch_low_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 12h Donchian midpoint
            if close[i] <= donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 12h Donchian midpoint
            if close[i] >= donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals