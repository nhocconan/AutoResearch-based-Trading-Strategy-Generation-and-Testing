#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_Regime_Change"
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
    
    # 1d data for TRIX, volume, and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate TRIX on 1d (15-period EMA of EMA of EMA)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3.diff() / ema3.shift(1)) * 100
    trix = trix.fillna(0).values
    
    # Volume spike on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    
    # Choppiness index on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_1d = pd.Series(np.maximum.reduce([
        high_1d[1:] - low_1d[1:],
        np.abs(high_1d[1:] - close_1d[:-1]),
        np.abs(low_1d[1:] - close_1d[:-1])
    ])).rolling(window=14, min_periods=14).mean().values
    # Prepend first value for alignment
    atr_1d = np.concatenate([[np.nan], atr_1d])
    
    true_range_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(true_range_sum / (max_hh - min_ll)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Align 1d indicators to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h EMA200 trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(ema200[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX turning up from negative, volume spike, chop < 61.8 (trending), above EMA200
            if (trix_aligned[i] > trix_aligned[i-1] and 
                trix_aligned[i] > 0 and 
                vol_spike_aligned[i] and 
                chop_aligned[i] < 61.8 and 
                close[i] > ema200[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX turning down from positive, volume spike, chop < 61.8 (trending), below EMA200
            elif (trix_aligned[i] < trix_aligned[i-1] and 
                  trix_aligned[i] < 0 and 
                  vol_spike_aligned[i] and 
                  chop_aligned[i] < 61.8 and 
                  close[i] < ema200[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX turns down OR chop > 61.8 (ranging) OR below EMA200
            if (trix_aligned[i] < trix_aligned[i-1] or 
                chop_aligned[i] > 61.8 or 
                close[i] < ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX turns up OR chop > 61.8 (ranging) OR above EMA200
            if (trix_aligned[i] > trix_aligned[i-1] or 
                chop_aligned[i] > 61.8 or 
                close[i] > ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals