#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Camarilla pivot levels (R3/S3) and volume confirmation.
# Long: Price crosses above S3 level (support) + volume > 1.5x 20-period average volume.
# Short: Price crosses below R3 level (resistance) + volume > 1.5x 20-period average volume.
# Uses Camarilla levels for mean-reversion in ranging markets and breakout detection in trends.
# Volume filter ensures participation. Position size 0.25 to balance risk and return.
# Target: 20-50 trades per year (~80-200 total over 4 years) to stay within fee limits.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Camarilla equations
        camarilla_s3[i] = pc - (ph - pl) * 1.1 / 6
        camarilla_r3[i] = pc + (ph - pl) * 1.1 / 6
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price crosses above S3 (support bounce) + volume confirmation
            if price > s3 and price <= s3 + (r3 - s3) * 0.05 and volume_confirm:  # near S3 with small buffer
                position = 1
                signals[i] = position_size
            # Short: price crosses below R3 (resistance rejection) + volume confirmation
            elif price < r3 and price >= r3 - (r3 - s3) * 0.05 and volume_confirm:  # near R3 with small buffer
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches midpoint or shows resistance
            midpoint = (r3 + s3) / 2
            if price >= midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches midpoint or shows support
            midpoint = (r3 + s3) / 2
            if price <= midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_R3S3_Volume"
timeframe = "4h"
leverage = 1.0