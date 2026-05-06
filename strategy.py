#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1/S1) with volume confirmation
# - Uses 1d Camarilla pivot levels for institutional support/resistance
# - Enters long when price closes above R1 with volume spike
# - Enters short when price closes below S1 with volume spike
# - Exits when price returns to pivot point (PP)
# - Uses 4h timeframe to balance trade frequency and signal quality
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_1dCamarilla_R1_S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pp = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels (R1 and S1 are most relevant for intraday trading)
    r1 = pp + (range_hl * 1.1 / 12)
    s1 = pp - (range_hl * 1.1 / 12)
    
    # Align 1d Camarilla levels to 4h timeframe (ONCE before loop)
    pp_4h = align_htf_to_ltf(prices, df_1d, pp)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter (4h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_4h[i]) or np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above R1 with volume spike
            if close[i] > r1_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below S1 with volume spike
            elif close[i] < s1_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point
            if close[i] < pp_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point
            if close[i] > pp_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals