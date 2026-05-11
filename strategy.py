#!/usr/bin/env python3
name = "4h_TrixVolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX on 1d for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix_raw = np.concatenate([[np.nan], trix_raw])
    trix = trix_raw
    trix_ma = pd.Series(trix).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_signal = trix - trix_ma
    
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Choppiness index on 1d for regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    atr1 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr1 = np.concatenate([[np.nan] * 13, atr1[13:]]) if len(atr1) > 13 else np.full_like(tr, np.nan)
    
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr1 / (hh - ll)) / np.log10(14)
    chop = np.concatenate([[np.nan] * 13, chop[13:]]) if len(chop) > 13 else np.full_like(high_1d, np.nan)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if np.isnan(trix_signal_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX signal crosses above 0, chop > 61.8 (range), volume spike
            if (trix_signal_aligned[i] > 0 and trix_signal_aligned[i-1] <= 0 and
                chop_aligned[i] > 61.8 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX signal crosses below 0, chop > 61.8 (range), volume spike
            elif (trix_signal_aligned[i] < 0 and trix_signal_aligned[i-1] >= 0 and
                  chop_aligned[i] > 61.8 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX signal crosses below 0 or chop < 38.2 (trend)
            if trix_signal_aligned[i] < 0 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX signal crosses above 0 or chop < 38.2 (trend)
            if trix_signal_aligned[i] > 0 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals