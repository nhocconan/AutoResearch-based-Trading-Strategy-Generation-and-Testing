#!/usr/bin/env python3
"""
12h_RSI_Reversal_With_1D_Volume_Filter
Hypothesis: On 12h timeframe, use RSI(14) oversold (<30) and overbought (>70) for mean-reversion signals, filtered by 1D volume spike (>1.5x 20-period average) to confirm institutional interest. Exit when RSI returns to neutral (40-60 range). This captures reversals in both bull and bear markets while avoiding low-probability choppy periods. Targets 15-25 trades/year with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    # Get 1D data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1D volume average (20-period)
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Volume spike: current volume > 1.5x 20-period average
    volume_spike_1d = np.where(vol_avg_1d > 0, volume_1d / vol_avg_1d, 0)
    
    # Align volume spike to 12h timeframe (wait for bar close)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume average and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI oversold (<30) and volume spike
            if (rsi[i] < 30 and volume_spike_1d_aligned[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) and volume spike
            elif (rsi[i] > 70 and volume_spike_1d_aligned[i] > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI returns to neutral (40-60)
            if 40 <= rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (40-60)
            if 40 <= rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_Reversal_With_1D_Volume_Filter"
timeframe = "12h"
leverage = 1.0