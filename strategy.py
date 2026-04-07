#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v2"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r3 = pivot + (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s3 = pivot - (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1)
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (24-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Price change momentum (2-period ROC on 12h)
    roc_2 = np.zeros(n)
    for i in range(2, n):
        roc_2[i] = (close[i] - close[i-2]) / close[i-2] if close[i-2] != 0 else 0
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(r3_12h[i]) or np.isnan(r4_12h[i]) or 
            np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(vol_ma[i]) or np.isnan(roc_2[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to Camarilla levels
        price_above_r3 = close[i] > r3_12h[i]
        price_below_s3 = close[i] < s3_12h[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Momentum confirmation
        mom_up = roc_2[i] > 0.003  # 0.3% momentum up
        mom_down = roc_2[i] < -0.003  # 0.3% momentum down
        
        # Exit conditions: opposite momentum or price reverses back to pivot
        pivot_12h = align_htf_to_ltf(prices, df_1d, (high_1d + low_1d + close_1d) / 3)
        exit_long = (roc_2[i] < -0.002) or (close[i] < pivot_12h[i])
        exit_short = (roc_2[i] > 0.002) or (close[i] > pivot_12h[i])
        
        if position == 1:  # Long position
            # Exit on negative momentum or price back to pivot
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on positive momentum or price back to pivot
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price above R3 + volume + upward momentum
            if price_above_r3 and vol_confirm and mom_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price below S3 + volume + downward momentum
            elif price_below_s3 and vol_confirm and mom_down:
                position = -1
                signals[i] = -0.25
    
    return signals