#!/usr/bin/env python3
name = "6H_Daily_IBS_EMA_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for IBS and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily IBS (Internal Bar Strength)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    range_1d = high_1d - low_1d
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    ibs = (close_1d - low_1d) / range_1d
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h
    ibs_aligned = align_htf_to_ltf(prices, df_1d, ibs)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ibs_aligned[i]) or np.isnan(ema50_aligned[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: IBS > 0.7 (strong close) + above daily EMA50 + volume confirmation
            if ibs_aligned[i] > 0.7 and close[i] > ema50_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: IBS < 0.3 (weak close) + below daily EMA50 + volume confirmation
            elif ibs_aligned[i] < 0.3 and close[i] < ema50_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: IBS < 0.3 (weak close) OR price below EMA50
            if ibs_aligned[i] < 0.3 or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: IBS > 0.7 (strong close) OR price above EMA50
            if ibs_aligned[i] > 0.7 or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals