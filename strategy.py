#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WK10_WMA_Cross_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for WMA crossover
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 5-week and 10-week WMA
    wma_5 = np.full_like(close_1w, np.nan)
    wma_10 = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i >= 4:  # 5-period WMA
            weights = np.arange(1, 6)
            wma_5[i] = np.dot(close_1w[i-4:i+1], weights) / weights.sum()
        if i >= 9:  # 10-period WMA
            weights = np.arange(1, 11)
            wma_10[i] = np.dot(close_1w[i-9:i+1], weights) / weights.sum()
    
    # Align WMAs to daily timeframe
    wma_5_aligned = align_htf_to_ltf(prices, df_1w, wma_5)
    wma_10_aligned = align_htf_to_ltf(prices, df_1w, wma_10)
    
    # Volume spike: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient data for WMA calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wma_5_aligned[i]) or np.isnan(wma_10_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: WMA5 crosses above WMA10 with volume spike
            long_cond = (wma_5_aligned[i] > wma_10_aligned[i] and 
                        wma_5_aligned[i-1] <= wma_10_aligned[i-1] and
                        volume_spike[i])
            
            # Short: WMA5 crosses below WMA10 with volume spike
            short_cond = (wma_5_aligned[i] < wma_10_aligned[i] and 
                         wma_5_aligned[i-1] >= wma_10_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: WMA5 crosses below WMA10
            if wma_5_aligned[i] < wma_10_aligned[i] and wma_5_aligned[i-1] >= wma_10_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: WMA5 crosses above WMA10
            if wma_5_aligned[i] > wma_10_aligned[i] and wma_5_aligned[i-1] <= wma_10_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals