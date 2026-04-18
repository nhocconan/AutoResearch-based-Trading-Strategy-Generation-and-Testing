#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S3_FadeWithVolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Previous weekly OHLC
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    # Weekly pivot and R3/S3 levels
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    range_w = prev_high_w - prev_low_w
    R3_w = pivot_w + range_w * 1.1
    S3_w = pivot_w - range_w * 1.1
    
    # R4/S4 for breakout confirmation (optional filter)
    R4_w = pivot_w + range_w * 1.6
    S4_w = pivot_w - range_w * 1.6
    
    # Align weekly levels to 6h
    R3_w_aligned = align_htf_to_ltf(prices, df_1w, R3_w)
    S3_w_aligned = align_htf_to_ltf(prices, df_1w, S3_w)
    R4_w_aligned = align_htf_to_ltf(prices, df_1w, R4_w)
    S4_w_aligned = align_htf_to_ltf(prices, df_1w, S4_w)
    
    # Volume filter: current volume > 1.5 * 48-period average (48 * 6h = 2 weeks)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_filter = volume > (1.5 * vol_ma_48)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R3_w_aligned[i]) or np.isnan(S3_w_aligned[i]) or
            np.isnan(R4_w_aligned[i]) or np.isnan(S4_w_aligned[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R3_val = R3_w_aligned[i]
        S3_val = S3_w_aligned[i]
        R4_val = R4_w_aligned[i]
        S4_val = S4_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Fade at R3/S3 with volume confirmation
            if close_val < R3_val and vol_filter:
                # Fade short at resistance
                signals[i] = -0.25
                position = -1
            elif close_val > S3_val and vol_filter:
                # Fade long at support
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Long exit: price reaches S4 (strong support) or closes back below S3
            if close_val <= S4_val or close_val < S3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches R4 (strong resistance) or closes back above R3
            if close_val >= R4_val or close_val > R3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals