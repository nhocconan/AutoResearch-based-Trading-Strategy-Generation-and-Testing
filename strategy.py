#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_Filter
Hypothesis: 6h Camarilla pivot breakout (R3/S3) with 1week trend filter (price >/<- EMA34 on weekly) and volume confirmation (>1.5x 20-bar avg). Enters long when price breaks above R3 in 1w uptrend, short when breaks below S3 in 1w downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 6h timeframe with ~12-30 trades/year, works in bull/bear by following 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivots for 1d
    # Pivot = (high + low + close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    R1_1d = pivot_1d + (range_1d * 1.0833 / 2)
    R2_1d = pivot_1d + (range_1d * 1.1666 / 2)
    R3_1d = pivot_1d + (range_1d * 1.2500 / 2)
    R4_1d = pivot_1d + (range_1d * 1.5000 / 2)
    
    # Support levels
    S1_1d = pivot_1d - (range_1d * 1.0833 / 2)
    S2_1d = pivot_1d - (range_1d * 1.1666 / 2)
    S3_1d = pivot_1d - (range_1d * 1.2500 / 2)
    S4_1d = pivot_1d - (range_1d * 1.5000 / 2)
    
    # Align Camarilla levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need sufficient data for EMA34 weekly
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in 1w uptrend with volume confirmation
            long_setup = (close[i] > R3_1d_aligned[i]) and (close_1w[i] > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short: price breaks below S3 in 1w downtrend with volume confirmation
            short_setup = (close[i] < S3_1d_aligned[i]) and (close_1w[i] < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S3 OR trend turns down
            if (close[i] < S3_1d_aligned[i]) or (close_1w[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R3 OR trend turns up
            if (close[i] > R3_1d_aligned[i]) or (close_1w[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0