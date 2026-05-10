#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v3
Hypothesis: Trade breakouts of daily Camarilla R3/S3 levels on 4h timeframe with 1d EMA34 trend filter and volume spike confirmation. 
Uses tighter entry conditions (volume > 2.5x 20 EMA) and longer minimum hold (4 bars) to reduce trade frequency and improve performance in both bull and bear markets.
Target: 20-35 trades/year.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (need previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 4h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2.5x 20-period EMA (tighter threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track bars since last entry to enforce min hold
    
    # Warmup: need 1d EMA (34) and Camarilla (need 2 days for shift)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                bars_since_entry = 0
            continue
        
        # Increment bars since entry if in a position
        if position != 0:
            bars_since_entry += 1
        
        # Trend filter: price vs 1d EMA34
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: break above R3 in uptrend with volume spike
            if high[i] > R3_aligned[i] and uptrend_1d and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below S3 in downtrend with volume spike
            elif low[i] < S3_aligned[i] and downtrend_1d and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Long exit: price drops below S3 or trend fails OR min hold met and reversal signal
            if bars_since_entry >= 4:
                # After minimum hold, exit on any reversal signal
                if low[i] < S3_aligned[i] or not uptrend_1d:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:
                # Before minimum hold, only exit on strong reversal
                if low[i] < S3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or trend fails OR min hold met and reversal signal
            if bars_since_entry >= 4:
                # After minimum hold, exit on any reversal signal
                if high[i] > R3_aligned[i] or not downtrend_1d:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
            else:
                # Before minimum hold, only exit on strong reversal
                if high[i] > R3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals