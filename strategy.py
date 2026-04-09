#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal + 1d ATR-based volatility filter + volume confirmation
# Williams %R identifies overbought/oversold conditions; 1d ATR filter ensures sufficient volatility for mean reversion to work
# Volume confirmation adds conviction to reversals. Works in ranging markets (2025-2026 bear/range) and captures mean reversion swings.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_williamsr_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.zeros(len(high_1d))
    if len(tr1) >= 14:
        atr_1d[13] = np.mean(tr1[:14])
        for i in range(14, len(tr1)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr1[i]) / 14
    
    # Align 1d ATR to 12h timeframe (wait for daily close)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if i < 13:
            williams_r[i] = np.nan
        else:
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high == lowest_low:
                williams_r[i] = -50.0  # avoid division by zero
            else:
                williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(avg_volume[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        # Volatility filter: 1d ATR > 20-period average ATR (ensures sufficient market movement)
        if i >= 20 + 100:  # enough history for ATR average
            atr_avg = np.mean(atr_1d_aligned[max(100, i-20):i])
            vol_filter = atr_1d_aligned[i] > atr_avg
        else:
            vol_filter = True  # allow trades during warmup of ATR average
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR loss of volume confirmation
            if williams_r[i] > -20 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR loss of volume confirmation
            if williams_r[i] < -80 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and volatility filter
            if volume_confirmed and vol_filter:
                # Long entry: Williams %R < -80 (oversold) - mean reversion long
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought) - mean reversion short
                elif williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals