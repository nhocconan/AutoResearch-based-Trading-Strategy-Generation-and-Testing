#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 breakout with volume confirmation and ATR filter.
# Long when price breaks above 12h R1 with volume > 2.0x 48-period average and ATR > 0.
# Short when price breaks below 12h S1 with volume > 2.0x 48-period average and ATR > 0.
# Exit when price crosses back below R1 (long) or above S1 (short).
# Uses 1d Camarilla pivot levels for structure, volume surge for conviction, ATR for volatility.
# Designed for ~12-25 trades/year per symbol.
name = "12h_1dPivot_R1S1_Breakout_VolumeATRFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align pivot levels to 12h timeframe (wait for daily bar to close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # ATR(14) on 12h for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 2.0 * 48-period average (48 * 12h = 24 days)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_filter = volume > (2.0 * vol_ma_48)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(atr_12h[i]) or np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        atr_val = atr_12h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and volatility
            if close_val > r1_val and vol_filter and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and volatility
            elif close_val < s1_val and vol_filter and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below R1
            if close_val < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above S1
            if close_val > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals