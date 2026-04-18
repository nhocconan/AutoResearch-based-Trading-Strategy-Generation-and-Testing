#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Pullback_Volume
4h strategy using daily Camarilla pivot R1/S1 levels with pullback to EMA34 and volume spike.
- Long: Price touches or crosses above S1 + pullback to EMA34 (bullish) + volume > 1.5x average
- Short: Price touches or crosses below R1 + pullback to EMA34 (bearish) + volume > 1.5x average
- Exit: Opposite signal or close beyond R3/S3
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in both bull and bear markets via mean-reversion at key pivot levels with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot points for daily timeframe
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate EMA34 on daily closes for pullback confirmation
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition (current volume > 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_avg[i]
        
        # Price touch conditions at S1/R1 with small tolerance
        touch_s1 = low[i] <= s1_aligned[i] * 1.001  # allow 0.1% overshoot
        touch_r1 = high[i] >= r1_aligned[i] * 0.999  # allow 0.1% undershoot
        
        # Pullback to EMA34 conditions
        pullback_bullish = close[i] > ema_34_aligned[i]  # price above EMA = bullish bias
        pullback_bearish = close[i] < ema_34_aligned[i]  # price below EMA = bearish bias
        
        # Exit conditions: price beyond R3/S3
        exit_long = high[i] >= r3_aligned[i] * 0.999
        exit_short = low[i] <= s3_aligned[i] * 1.001
        
        if position == 0:
            # Long: touch S1 + bullish pullback + volume spike
            if touch_s1 and pullback_bullish and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: touch R1 + bearish pullback + volume spike
            elif touch_r1 and pullback_bearish and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: touch R1 (reverse) or exit beyond R3
            if touch_r1 or exit_long:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: touch S1 (reverse) or exit beyond S3
            if touch_s1 or exit_short:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Pullback_Volume"
timeframe = "4h"
leverage = 1.0