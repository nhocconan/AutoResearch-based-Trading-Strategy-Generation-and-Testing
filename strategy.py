#!/usr/bin/env python3
name = "4h_TrixVolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for TRIX, volume spike, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX (15-period triple EMA) on 1d
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value undefined
    
    # Align TRIX to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index on 1d (14-period)
    atr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = tr if i < 14 else (atr_1d[i-1] * 13 + tr) / 14
    atr_1d[0] = high_1d[0] - low_1d[0]
    
    max_high = np.maximum.accumulate(high_1d)
    min_low = np.minimum.accumulate(low_1d)
    range_max_h = max_high - min_low
    chop = np.zeros(len(close_1d))
    for i in range(14, len(close_1d)):
        sum_atr = np.sum(atr_1d[i-13:i+1])
        if range_max_h[i] > 0:
            chop[i] = 100 * np.log10(sum_atr / range_max_h[i]) / np.log10(14)
        else:
            chop[i] = 50
    chop[:14] = 50
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above 0, volume spike, chop < 61.8 (trending)
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                vol_spike[i] and chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below 0, volume spike, chop < 61.8 (trending)
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  vol_spike[i] and chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below 0 or chop > 61.8 (range)
            if trix_aligned[i] < 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above 0 or chop > 61.8 (range)
            if trix_aligned[i] > 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals