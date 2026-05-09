#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ChopFilter_Trix"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter and chop calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on 1w close
    close_1w = df_1w['close'].values
    # TRIX: triple EMA of log returns
    log_ret = np.diff(np.log(np.concatenate([[close_1w[0]], close_1w])))
    ema1 = pd.Series(log_ret).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * np.diff(ema3) / ema3[:-1]  # percentage change
    trix = np.concatenate([[0], trix])  # align length
    trix = np.concatenate([[0], trix])  # second diff for triple
    trix_1w = np.concatenate([[0, 0], trix[2:]])  # final alignment
    trix_1w_aligned = align_htf_to_ltf(prices, df_1w, trix_1w)
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_1w).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_signal_aligned = align_htf_to_ltf(prices, df_1w, trix_signal)
    
    # Calculate 1w Choppiness Index
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w_arr[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w_arr[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0], tr])  # align to same length
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr = np.concatenate([np.zeros(13), atr[13:]])  # align
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    tr_sum = np.concatenate([np.zeros(13), tr_sum[13:]])
    
    # Max/min close over 14 periods
    max_close = pd.Series(close_1w_arr).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1w_arr).rolling(window=14, min_periods=14).min().values
    max_close = np.concatenate([np.zeros(13), max_close[13:]])
    min_close = np.concatenate([np.zeros(13), min_close[13:]])
    
    # Chop = 100 * log10(tr_sum / (max_close - min_close)) / log10(14)
    range_14 = max_close - min_close
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(tr_sum / range_14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(trix_1w_aligned[i]) or 
            np.isnan(trix_signal_aligned[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_1w_aligned[i]
        trix_signal_val = trix_signal_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Enter long: TRIX crosses above signal line AND chop < 61.8 (trending market)
            if trix_val > trix_signal_val and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below signal line AND chop < 61.8 (trending market)
            elif trix_val < trix_signal_val and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line OR chop > 61.8 (ranging market)
            if trix_val < trix_signal_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line OR chop > 61.8 (ranging market)
            if trix_val > trix_signal_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals