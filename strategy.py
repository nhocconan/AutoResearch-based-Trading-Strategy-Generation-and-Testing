#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Trix_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX on 1w (weekly) as primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA1
    ema1 = pd.Series(close_1w).ewm(span=9, adjust=False, min_periods=9).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix_raw[0] = 0
    # Smooth TRIX with 9-period EMA
    trix = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_1w_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # Calculate Chop Index on 1d for regime detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First value
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Absolute close change over 14 periods
    abs_close_change = np.abs(close_1d - np.roll(close_1d, 14))
    abs_close_change[:14] = 0
    # Sum of absolute close changes
    abs_close_sum = pd.Series(abs_close_change).rolling(window=14, min_periods=14).sum().values
    
    # Chop Index = 100 * log10(tr_sum / abs_close_sum) / log10(14)
    chop = np.zeros_like(tr_sum)
    mask = (tr_sum > 0) & (abs_close_sum > 0)
    chop[mask] = 100 * np.log10(tr_sum[mask] / abs_close_sum[mask]) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX > 0 (uptrend) + Chop > 61.8 (ranging) + volume spike
            long_cond = (trix_1w_aligned[i] > 0) and \
                        (chop_1d_aligned[i] > 61.8) and \
                        volume_spike[i]
            # Short: TRIX < 0 (downtrend) + Chop > 61.8 (ranging) + volume spike
            short_cond = (trix_1w_aligned[i] < 0) and \
                         (chop_1d_aligned[i] > 61.8) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative OR Chop < 38.2 (trending)
            if (trix_1w_aligned[i] < 0) or (chop_1d_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive OR Chop < 38.2 (trending)
            if (trix_1w_aligned[i] > 0) or (chop_1d_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals