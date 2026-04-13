#!/usr/bin/env python3
"""
1D Weekly CCI Trend with Daily Volume Confirmation.
Uses weekly CCI to identify strong trends, enters on daily pullbacks with volume spikes.
Designed to work in both bull and bear markets by following the weekly trend direction.
Targets 30-100 trades over 4 years with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get daily data for entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for CCI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly CCI (20-period)
    tp_1w = (high_1w + low_1w + close_1w) / 3
    sma_tp = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_20 = (tp_1w - sma_tp) / (0.015 * mad)
    
    # Trend: CCI > 100 = strong uptrend, CCI < -100 = strong downtrend
    strong_uptrend = cci_20 > 100
    strong_downtrend = cci_20 < -100
    
    # Align weekly trend to daily
    strong_uptrend_aligned = align_htf_to_ltf(prices, df_1w, strong_uptrend.astype(float))
    strong_downtrend_aligned = align_htf_to_ltf(prices, df_1w, strong_downtrend.astype(float))
    
    # Daily volume confirmation
    vol_ma_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_spike = volume_1d > (vol_ma_10 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Daily pullback entry: price near 20-period EMA in trend direction
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    pullback_long = close_1d <= ema_20 * 1.02  # Within 2% above EMA
    pullback_short = close_1d >= ema_20 * 0.98  # Within 2% below EMA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(strong_uptrend_aligned[i]) or 
            np.isnan(strong_downtrend_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: strong weekly trend + volume spike + daily pullback
        long_entry = (strong_uptrend_aligned[i] > 0.5 and 
                      vol_spike_aligned[i] > 0.5 and 
                      pullback_long)
        short_entry = (strong_downtrend_aligned[i] > 0.5 and 
                       vol_spike_aligned[i] > 0.5 and 
                       pullback_short)
        
        # Exit when trend weakens (CCI returns to normal range)
        exit_long = position == 1 and strong_uptrend_aligned[i] < 0.5
        exit_short = position == -1 and strong_downtrend_aligned[i] < 0.5
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_weekly_cci_volume_pullback"
timeframe = "1d"
leverage = 1.0