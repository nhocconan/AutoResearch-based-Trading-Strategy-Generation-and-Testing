#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TrixVolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for TRIX and chop regime
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate TRIX on 1h close
    close_1h = df_1h['close'].values
    # EMA1
    ema1 = pd.Series(close_1h).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / np.where(np.abs(ema3[:-1]) > 1e-8, np.abs(ema3[:-1]), 1e-8) * 100
    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Align TRIX signal to 4h
    trix_signal_aligned = align_htf_to_ltf(prices, df_1h, trix_signal)
    
    # Calculate Choppiness Index on 1h (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # ATR14
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of ATR14 over 14 periods
    atr_sum = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    # Max and min close over 14 periods
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = LOG10(atr_sum / (max_hh - min_ll)) / LOG10(14) * 100
    range_hl = max_hh - min_ll
    chop = np.zeros_like(atr_sum)
    mask = (range_hl > 0) & (~np.isnan(atr_sum))
    chop[mask] = (np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(14)) * 100
    chop = np.nan_to_num(chop, nan=50.0)
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1h, chop)
    
    # Volume spike on 4h: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_spike = np.nan_to_num(vol_spike, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trix_signal_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line + chop > 61.8 (ranging) + volume spike
            if (trix_signal_aligned[i] > 0 and 
                trix_signal_aligned[i-1] <= 0 and
                chop_aligned[i] > 61.8 and
                vol_spike[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line + chop > 61.8 (ranging) + volume spike
            elif (trix_signal_aligned[i] < 0 and 
                  trix_signal_aligned[i-1] >= 0 and
                  chop_aligned[i] > 61.8 and
                  vol_spike[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero OR chop < 38.2 (trending)
            if trix_signal_aligned[i] < 0 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero OR chop < 38.2 (trending)
            if trix_signal_aligned[i] > 0 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals