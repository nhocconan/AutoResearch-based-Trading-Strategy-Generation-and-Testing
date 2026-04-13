#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h Camarilla pivot levels and volume confirmation.
# Long: Price breaks above 12h R3 level + volume > 1.3x average volume (20-period).
# Short: Price breaks below 12h S3 level + volume > 1.3x average volume.
# Uses 12h for Camarilla pivot structure, 6h for execution with volume confirmation.
# Position size: 0.25. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (using previous period's HLC)
    camarilla_r3 = np.full(len(high_12h), np.nan)
    camarilla_s3 = np.full(len(high_12h), np.nan)
    
    for i in range(1, len(high_12h)):
        # Previous period's high, low, close
        ph = high_12h[i-1]
        pl = low_12h[i-1]
        pc = close_12h[i-1]
        
        # Camarilla formulas
        range_ = ph - pl
        camarilla_r3[i] = pc + range_ * 1.1 / 2
        camarilla_s3[i] = pc - range_ * 1.1 / 2
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 12h Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: break above R3 + volume confirmation
            if price > r3 and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: break below S3 + volume confirmation
            elif price < s3 and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below S3 (reversion to mean)
            if price < s3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above R3 (reversion to mean)
            if price > r3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Camarilla_R3S3_Volume"
timeframe = "6h"
leverage = 1.0