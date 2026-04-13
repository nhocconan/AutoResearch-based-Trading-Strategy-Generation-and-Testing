#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation.
# Camarilla levels from 12h data provide institutional support/resistance.
# Breakout at R4/S4 with volume > 1.5x 12h average volume confirms institutional interest.
# Works in bull/bear as it captures breakouts in either direction.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla levels and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    camarilla_r4 = np.zeros(len(close_12h))
    camarilla_s4 = np.zeros(len(close_12h))
    
    for i in range(1, len(close_12h)):
        hl_range = high_12h[i-1] - low_12h[i-1]
        camarilla_r4[i] = close_12h[i-1] + hl_range * 1.1 / 2
        camarilla_s4[i] = close_12h[i-1] - hl_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (will use previous 12h bar's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Calculate 12h average volume (20-period) for volume confirmation
    vol_12h = df_12h['volume'].values
    avg_vol_12h = np.zeros(len(vol_12h))
    for i in range(20, len(vol_12h)):
        avg_vol_12h[i] = np.mean(vol_12h[i-20:i])
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(avg_vol_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        avg_vol = avg_vol_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average 12h volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above R4 with volume confirmation
            if price > r4 and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S4 with volume confirmation
            elif price < s4 and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below R4 (failed breakout)
            if price < r4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns above S4 (failed breakout)
            if price > s4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Camarilla_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0