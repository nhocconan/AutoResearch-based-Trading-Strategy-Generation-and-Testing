#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_Regime"
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
    
    # TRIX on 4h close (12-period EMA smoothed three times)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # 1d data for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Volume spike: current volume > 2x 20-day average
    vol_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (vol_20d * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Choppiness Index (14-period) for regime filter
    atr_14 = []
    for i in range(len(df_1d)):
        if i < 14:
            atr_14.append(np.nan)
        else:
            tr = np.max([
                df_1d['high'].values[i] - df_1d['low'].values[i],
                abs(df_1d['high'].values[i] - df_1d['close'].values[i-1]),
                abs(df_1d['low'].values[i] - df_1d['close'].values[i-1])
            ])
            atr_14.append(tr)
    atr_14 = np.array(atr_14)
    
    sum_tr14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr14 / (highest_high - lowest_low)) / np.log10(14)
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Ensure TRIX and chop are ready
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_spike_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal, chop < 61.8 (trending), volume spike
            if (trix[i] > trix_signal[i] and 
                trix[i-1] <= trix_signal[i-1] and 
                chop_aligned[i] < 61.8 and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal, chop < 61.8 (trending), volume spike
            elif (trix[i] < trix_signal[i] and 
                  trix[i-1] >= trix_signal[i-1] and 
                  chop_aligned[i] < 61.8 and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below signal or chop > 61.8 (ranging)
            if trix[i] < trix_signal[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above signal or chop > 61.8 (ranging)
            if trix[i] > trix_signal[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals