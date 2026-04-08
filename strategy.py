#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_volume_v1
# Hypothesis: 4h price breaking above/below daily Camarilla pivot point resistance/support levels
# (R4/S4) with volume confirmation creates high-probability breakout trades in both bull and bear markets.
# Uses daily timeframe for pivot calculation (major support/resistance) and 4h for entry timing.
# Volume filter ensures breakouts have conviction. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    # R3/S3 for exit: R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4)
    r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.5x average of last 12 periods (6 hours)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or \
           np.isnan(s4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (reversal signal)
            if close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (reversal signal)
            if close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with volume
            if (close[i] > r4_1d_aligned[i] and 
                open_prices[i] <= r4_1d_aligned[i] and  # Ensure breakout happened this bar
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S4 with volume
            elif (close[i] < s4_1d_aligned[i] and 
                  open_prices[i] >= s4_1d_aligned[i] and  # Ensure breakdown happened this bar
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals