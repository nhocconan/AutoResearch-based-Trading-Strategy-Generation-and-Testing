#!/usr/bin/env python3
name = "4h_TrixVolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for TRIX and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # TRIX (15-period EMA of EMA of EMA of close, then % change)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix[0] = 0.0
    
    # ATR for regime detection (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align TRIX and TRIX signal to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(15, 9, 14, 20)
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or np.isnan(atr_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal + volume spike
            if trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal + volume spike
            elif trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below signal
            if trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above signal
            if trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals