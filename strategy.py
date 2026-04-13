#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Camarilla pivot levels and volume confirmation.
# Long: Price crosses above Camarilla R3 (1d) + volume > 1.5x average volume (20-period).
# Short: Price crosses below Camarilla S3 (1d) + volume > 1.5x average volume.
# Uses 1d pivot levels for key support/resistance, volume for confirmation.
# Position size: 0.25 (25% of capital).
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(low_1d), np.nan)
    for i in range(len(high_1d)):
        # Camarilla formulas
        range_ = high_1d[i] - low_1d[i]
        camarilla_r3[i] = close_1d[i] + range_ * 1.1 / 6
        camarilla_s3[i] = close_1d[i] - range_ * 1.1 / 6
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Camarilla levels to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: cross above R3 + volume confirmation
            if price > r3 and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: cross below S3 + volume confirmation
            elif price < s3 and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below S3 (opposite level)
            if price < s3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above R3 (opposite level)
            if price > r3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_R3S3_Volume"
timeframe = "12h"
leverage = 1.0