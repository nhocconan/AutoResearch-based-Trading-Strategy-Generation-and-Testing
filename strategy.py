#!/usr/bin/env python3
name = "12h_TRIX_Volume_Spike_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX on 1D: EMA15 of EMA15 of EMA15
    ema1 = pd.Series(df_1d['close']).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX signal
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Volume spike: volume > 2x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # 1D EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema34_aligned[i]) or np.isnan(trix_signal_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal, price > EMA34, volume spike
            if trix_signal_aligned[i] > trix_signal_aligned[i-1] and \
               trix_signal_aligned[i-1] <= trix_signal_aligned[i-2] and \
               close[i] > ema34_aligned[i] and \
               volume[i] > 2 * vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal, price < EMA34, volume spike
            elif trix_signal_aligned[i] < trix_signal_aligned[i-1] and \
                 trix_signal_aligned[i-1] >= trix_signal_aligned[i-2] and \
                 close[i] < ema34_aligned[i] and \
                 volume[i] > 2 * vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal or price < EMA34
            if trix_signal_aligned[i] < trix_signal_aligned[i-1] and \
               trix_signal_aligned[i-1] >= trix_signal_aligned[i-2] or \
               close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal or price > EMA34
            if trix_signal_aligned[i] > trix_signal_aligned[i-1] and \
               trix_signal_aligned[i-1] <= trix_signal_aligned[i-2] or \
               close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals