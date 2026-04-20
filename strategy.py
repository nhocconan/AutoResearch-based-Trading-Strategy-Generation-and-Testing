#!/usr/bin/env python3
# 1d_1w_WeeklyPivot_Breakout_Volume
# Hypothesis: On daily timeframe, trade breakouts from weekly-derived Camarilla R1/S1 levels with volume spike confirmation.
# Uses weekly pivot points (R1/S1) as key support/resistance levels. Breaks above R1 or below S1 with volume > 2x 20-day average
# indicate institutional interest and potential trend continuation. Works in both bull and bear markets by trading
# breakouts in the direction of the weekly trend implied by price relative to weekly pivot.
# Targets 10-25 trades per year to minimize fee drag while capturing significant moves.

name = "1d_1w_WeeklyPivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R1, S1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    
    # Pivot point and ranges
    pivot_1w = typical_price_1w
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R1 and S1 (using 1.1 multiplier / 2 for R1/S1)
    s1_1w = close_1w - (range_1w * 1.1 / 2)
    r1_1w = close_1w + (range_1w * 1.1 / 2)
    
    # Align weekly levels to daily timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R1, volume spike
            if (close[i] > r1_aligned[i] * 1.002 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S1, volume spike
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 (reversal to opposite level)
            if close[i] < s1_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R1 (reversal to opposite level)
            if close[i] > r1_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals