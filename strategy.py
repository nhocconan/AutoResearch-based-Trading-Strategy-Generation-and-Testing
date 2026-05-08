#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_Volume_Spike_Chop_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # TRIX: Triple EMA of percentage change
    # EMA1 of close
    ema1 = pd.Series(df_1d['close']).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (EMA3_today - EMA3_yesterday) / EMA3_yesterday
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix_raw[0] = 0
    
    # Align TRIX to 4h timeframe
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    
    # Chop filter on weekly: Choppy if > 61.8 (range), Trending if < 38.2
    # Calculate True Range and ATR for chop
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Chop = 100 * log10(tr_sum_14 / (atr_1w * 14)) / log10(14)
    chop_raw = 100 * np.log10(tr_sum_14 / (atr_1w * 14)) / np.log10(14)
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)
    
    # Align chop to 4h timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_raw)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + chop > 61.8 (range) + volume spike
            if (trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0 and
                chop_1w_aligned[i] > 61.8 and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + chop > 61.8 (range) + volume spike
            elif (trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0 and
                  chop_1w_aligned[i] > 61.8 and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero OR chop < 38.2 (trend)
            if trix_1d_aligned[i] < 0 or chop_1w_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero OR chop < 38.2 (trend)
            if trix_1d_aligned[i] > 0 or chop_1w_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals