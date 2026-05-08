#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Vortex_Trend_Filter_1wVolatility"
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
    
    # Weekly data for volatility filter (Vortex needs ATR)
    df_1w = get_htf_1w_data(prices)
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Daily data for Vortex trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1w (for volatility filter)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate True Range for 1d (for Vortex)
    tr1_d = high_1d[1:] - low_1d[1:]
    tr2_d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    
    # Calculate +VM and -VM for Vortex
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    
    # Sum over 14 periods
    sum_vm_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    sum_tr = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    vi_plus = np.where(sum_tr != 0, sum_vm_plus / sum_tr, 0)
    vi_minus = np.where(sum_tr != 0, sum_vm_minus / sum_tr, 0)
    
    # Vortex indicators: VI+ > VI- = bullish, VI- > VI+ = bearish
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # Volatility filter: only trade when 1w ATR is above its 50-period average (avoid low volatility chop)
    atr_ma_50 = pd.Series(atr_1w_aligned).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_1w_aligned > atr_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ > VI- and volatility filter
            long_cond = (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                        volatility_filter[i])
            
            # Short: VI- > VI+ and volatility filter
            short_cond = (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                         volatility_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VI- crosses above VI+
            if vi_minus_aligned[i] > vi_plus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: VI+ crosses above VI-
            if vi_plus_aligned[i] > vi_minus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def get_htf_1w_data(prices):
    """Get 1h data by calling get_htf_data with '1w'"""
    return get_htf_data(prices, '1w')