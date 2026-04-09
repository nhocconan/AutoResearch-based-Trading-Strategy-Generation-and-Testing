#!/usr/bin/env python3
# 4h_camarilla_volume_breakout_v1
# Hypothesis: Uses daily Camarilla pivot levels (from 1d) with 4h breakouts and volume confirmation.
# In bull markets: buy breakout above H3; in bear markets: sell breakdown below L3.
# Volume > 1.5x 20-period average confirms breakout strength.
# Works in both bull and bear by trading breakouts in direction of higher timeframe trend.
# Target: 20-40 trades/year (80-160 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_volume_breakout_v1"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    # H3 = C + (H-L)*1.1/2, L3 = C - (H-L)*1.1/2
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 or volume fails
            if close[i] < L3_4h[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above H3 or volume fails
            if close[i] > H3_4h[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume
            if close[i] > H3_4h[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume
            elif close[i] < L3_4h[i] and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals