#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_ChoppyRegime"
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
    
    # === TRIX: TRIPLE EMA OF LOG RETURNS ===
    # Calculate log returns
    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    # Triple EMA of log returns
    ema1 = pd.Series(log_returns).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)  # % change
    trix[0] = 0  # first value
    
    # === TRIX SIGNAL LINE (EMA OF TRIX) ===
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # === 1D DATA FOR CHOPPINESS INDEX ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Choppy Index: (sum(TR,14) / (max(high,14) - min(low,14))) * 100
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 / (max_high14 - min_low14)) / np.log10(14)
    chop = np.where((max_high14 - min_low14) == 0, 50, chop)  # avoid division by zero
    
    # Align TRIX and Chop to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_4h = align_htf_to_ltf(prices, df_1d, trix_signal)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # Strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_4h[i]) or np.isnan(trix_signal_4h[i]) or 
            np.isnan(chop_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above signal + choppy regime (range) + volume spike
            if (trix_4h[i] > trix_signal_4h[i] and trix_4h[i-1] <= trix_signal_4h[i-1] and
                chop_4h[i] > 50 and  # choppy/ranging market
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below signal + choppy regime + volume spike
            elif (trix_4h[i] < trix_signal_4h[i] and trix_4h[i-1] >= trix_signal_4h[i-1] and
                  chop_4h[i] > 50 and  # choppy/ranging market
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal OR chop becomes too low (trending)
            if trix_4h[i] < trix_signal_4h[i] or chop_4h[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal OR chop becomes too low (trending)
            if trix_4h[i] > trix_signal_4h[i] or chop_4h[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals