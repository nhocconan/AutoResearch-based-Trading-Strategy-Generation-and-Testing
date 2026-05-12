#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_ChoppyRegime
Hypothesis: TRIX momentum with volume spike and Choppiness regime filter captures trend changes in 12h timeframe. TRIX filters noise, volume confirms conviction, Choppiness > 61.8 signals ranging (mean-reversion) while < 38.2 signals trending (trend-follow). Works in bull/bear by adapting to regime: mean revert in chop, follow trend in trend. Target: 12-37 trades/year per symbol.
"""

name = "12h_TRIX_VolumeSpike_ChoppyRegime"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # TRIX on 1d: triple EMA of log returns
    close_1d = df_1d['close'].values
    log_returns = np.diff(np.log(close_1d), prepend=np.log(close_1d[0]))
    ema1 = pd.Series(log_returns).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0
    
    # TRIX signal line (9-period EMA)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal to 12h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Choppiness Index on 1d
    atr_1d = np.zeros(len(close_1d))
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (max_hh - min_ll)) / np.log10(14)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)
    
    # Align Choppiness to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: >1.8x 30-period average (12h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above signal + chop < 38.2 (trending) + volume spike
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                chop_aligned[i] < 38.2 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below signal + chop < 38.2 (trending) + volume spike
            elif (trix_aligned[i] < trix_signal_aligned[i] and 
                  trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                  chop_aligned[i] < 38.2 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal OR chop > 61.8 (choppy) 
            if (trix_aligned[i] < trix_signal_aligned[i] and 
                trix_aligned[i-1] >= trix_signal_aligned[i-1]) or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal OR chop > 61.8 (choppy)
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1]) or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals