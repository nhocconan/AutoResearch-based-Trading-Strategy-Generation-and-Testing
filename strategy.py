#!/usr/bin/env python3
name = "6h_SMI_Trend_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _smi(high, low, close, k=10, d=3, smooth=3):
    """Calculate Stochastic Momentum Index (SMI)"""
    n = len(close)
    if n < k:
        return np.full(n, np.nan)
    
    # Calculate hlc3 (midpoint of high-low)
    hlc3 = (high + low + close) / 3.0
    
    # Calculate min and max of hlc3 over k periods
    min_hlc3 = pd.Series(hlc3).rolling(window=k, min_periods=k).min().values
    max_hlc3 = pd.Series(hlc3).rolling(window=k, min_periods=k).max().values
    
    # Avoid division by zero
    range_hlc3 = max_hlc3 - min_hlc3
    range_hlc3 = np.where(range_hlc3 == 0, 1e-10, range_hlc3)
    
    # Calculate SMI raw value
    smi_raw = (hlc3 - (min_hlc3 + max_hlc3) / 2.0) / (range_hlc3 / 2.0) * 100.0
    
    # Smooth with EMA (double smoothed)
    smi_once = pd.Series(smi_raw).ewm(span=smooth, adjust=False).mean().values
    smi_twice = pd.Series(smi_once).ewm(span=smooth, adjust=False).mean().values
    
    # Signal line
    smi_signal = pd.Series(smi_twice).ewm(span=d, adjust=False).mean().values
    
    return smi_twice, smi_signal

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (using 1w EMA20)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get 1d data for SMI calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate SMI on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    smi, smi_signal = _smi(high_1d, low_1d, close_1d, k=10, d=3, smooth=3)
    smi_aligned = align_htf_to_ltf(prices, df_1d, smi)
    smi_signal_aligned = align_htf_to_ltf(prices, df_1d, smi_signal)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(smi_aligned[i]) or np.isnan(smi_signal_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: SMI crosses above signal line AND above 1w EMA20 (uptrend) AND volume surge
            if smi_aligned[i] > smi_signal_aligned[i] and smi_aligned[i-1] <= smi_signal_aligned[i-1] and close[i] > ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: SMI crosses below signal line AND below 1w EMA20 (downtrend) AND volume surge
            elif smi_aligned[i] < smi_signal_aligned[i] and smi_aligned[i-1] >= smi_signal_aligned[i-1] and close[i] < ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: SMI crosses below signal line OR below 1w EMA20 (trend change)
            if smi_aligned[i] < smi_signal_aligned[i] and smi_aligned[i-1] >= smi_signal_aligned[i-1] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: SMI crosses above signal line OR above 1w EMA20 (trend change)
            if smi_aligned[i] > smi_signal_aligned[i] and smi_aligned[i-1] <= smi_signal_aligned[i-1] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals