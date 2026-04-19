#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_MultiTimeframe_Momentum_Volume_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for momentum and volume calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d momentum (close - close 10 days ago)
    close_1d = df_1d['close'].values
    momentum_1d = np.full(len(close_1d), np.nan)
    for i in range(10, len(close_1d)):
        momentum_1d[i] = close_1d[i] - close_1d[i-10]
    
    # Calculate 1d volume average
    volume_1d = df_1d['volume'].values
    volume_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        volume_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Calculate 6h volume spike (volume > 1.5 * 20-period average)
    volume_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        volume_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (volume_ma_6h * 1.5)
    
    # Align 1d indicators to 6h timeframe
    momentum_1d_aligned = align_htf_to_ltf(prices, df_1d, momentum_1d)
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(momentum_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Momentum signals:
        # Positive momentum: bullish, Negative momentum: bearish
        # Volume confirmation required
        vol_confirm = volume_spike_6h[i]
        
        if position == 0:
            # Long when positive momentum + volume spike
            if momentum_1d_aligned[i] > 0 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when negative momentum + volume spike
            elif momentum_1d_aligned[i] < 0 and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when momentum turns negative
            if momentum_1d_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when momentum turns positive
            if momentum_1d_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals