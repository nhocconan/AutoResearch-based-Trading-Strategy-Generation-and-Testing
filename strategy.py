#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX on close (12-period EMA of EMA of EMA)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # percentage change
    trix_values = trix.values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Choppiness Index (14-period) for regime filter
    atr = np.abs(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))))
    atr[0] = np.abs(high[0] - low[0])  # first value
    tr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (np.log14(14) * (highest_high - lowest_low))) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) != 0, chop, 50.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Daily trend filter using EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_values[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and chop < 61.8 (trending) and daily uptrend
            long_cond = (trix_values[i] > 0 and trix_values[i-1] <= 0 and
                        volume_spike[i] and chop[i] < 61.8 and
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1])
            
            # Short: TRIX crosses below zero with volume spike and chop < 61.8 (trending) and daily downtrend
            short_cond = (trix_values[i] < 0 and trix_values[i-1] >= 0 and
                         volume_spike[i] and chop[i] < 61.8 and
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_values[i] < 0 and trix_values[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_values[i] > 0 and trix_values[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def np_log10(x):
    return np.log(x) / np.log(10)