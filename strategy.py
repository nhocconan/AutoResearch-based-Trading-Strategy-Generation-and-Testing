#!/usr/bin/env python3
# 12h_camilla_pivot_breakout_volume_v1
# Hypothesis: Camarilla pivot breakouts with volume confirmation on 12h timeframe.
# Long when price breaks above S3 pivot with volume > 1.5x average volume.
# Short when price breaks below S4 pivot with volume > 1.5x average volume.
# Exit when price returns to the 50% level (midpoint between S3 and S4).
# Uses daily pivots for structure, 12h for execution, volume for confirmation.
# Target: 15-30 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camilla_pivot_breakout_volume_v1"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (based on previous day's range)
    # S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use the previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot levels
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    mid = (s3 + s4) / 2  # 50% level for exit
    
    # Align to 12h timeframe (wait for daily bar to close)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    mid_aligned = align_htf_to_ltf(prices, df_1d, mid)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(mid_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to 50% level (midpoint)
            if close[i] <= mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to 50% level (midpoint)
            if close[i] >= mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above S3 with volume surge
            if (close[i] > s3_aligned[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S4 with volume surge
            elif (close[i] < s4_aligned[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals