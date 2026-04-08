#!/usr/bin/env python3
# 12h_1d_camarilla_volume_v1
# Hypothesis: Trade Camarilla pivot level reversals on 12h with 1d volume confirmation.
# Enter long when price bounces off S3 support with 1d volume surge.
# Enter short when price is rejected at R3 resistance with 1d volume surge.
# Exit when price reaches opposite pivot level or volume drops.
# Camarilla levels provide institutional reference points; volume confirms institutional interest.
# Target: 15-25 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    camarilla_r3 = np.zeros(len(high_1d))
    camarilla_s3 = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        H = high_1d[i]
        L = low_1d[i]
        C = close_1d[i]
        range_hl = H - L
        
        camarilla_r3[i] = C + (range_hl * 1.2500)
        camarilla_s3[i] = C - (range_hl * 1.2500)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 1d volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition (using current 12h bar volume vs 20-period MA)
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price reaches S3 level or volume drops significantly
            if low[i] <= camarilla_s3_aligned[i] or not vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches R3 level or volume drops significantly
            if high[i] >= camarilla_r3_aligned[i] or not vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price touches/bounces off S3 with volume surge
            if (low[i] <= camarilla_s3_aligned[i] * 1.002 and  # Allow small buffer
                close[i] > camarilla_s3_aligned[i] and        # Price closes above S3
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches/rejected at R3 with volume surge
            elif (high[i] >= camarilla_r3_aligned[i] * 0.998 and  # Allow small buffer
                  close[i] < camarilla_r3_aligned[i] and        # Price closes below R3
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals