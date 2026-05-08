#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_TRIX_Crossover_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d TRIX (15-period EMA of EMA of EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 45:  # Need enough for triple EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value undefined
    
    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Align all 1d indicators to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line, price above 1d EMA34, volume spike
            long_cond = (trix_aligned[i] > trix_signal_aligned[i] and 
                        trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                        close[i] > ema34_1d_aligned[i] and
                        volume_spike[i])
            
            # Short: TRIX crosses below signal line, price below 1d EMA34, volume spike
            short_cond = (trix_aligned[i] < trix_signal_aligned[i] and 
                         trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                         close[i] < ema34_1d_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line OR price crosses below 1d EMA34
            if (trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1]) or \
               close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line OR price crosses above 1d EMA34
            if (trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1]) or \
               close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals