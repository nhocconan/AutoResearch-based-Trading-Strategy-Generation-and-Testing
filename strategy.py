#!/usr/bin/env python3
"""
12h_HMA_Crossover_1DTrend_Volume
Hypothesis: Hull Moving Average crossover (16/32) with 1d trend filter and volume confirmation works in both bull and bear markets.
Long: HMA(16) crosses above HMA(32) with 1d uptrend and volume spike.
Short: HMA(16) crosses below HMA(32) with 1d downtrend and volume spike.
Exit on opposite crossover. Uses volume > 2x 20-period average for confirmation.
Target: 15-30 trades/year per symbol to minimize fee drag.
"""

name = "12h_HMA_Crossover_1DTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def weighted_moving_average(array, window):
    """Calculate weighted moving average with weights 1,2,3,...,window"""
    weights = np.arange(1, window + 1)
    return np.convolve(array, weights, 'full')[:len(array)] / weights.sum()

def hull_moving_average(array, period):
    """Calculate Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = weighted_moving_average(array, half_period)
    wma_full = weighted_moving_average(array, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = weighted_moving_average(raw_hma, sqrt_period)
    
    # Handle NaN values from convolution
    hma = np.where(np.isnan(hma), 0, hma)
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # HMA indicators
    hma_fast = hull_moving_average(close, 16)
    hma_slow = hull_moving_average(close, 32)
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after HMA warmup
        # Get values
        hma_f = hma_fast[i]
        hma_s = hma_slow[i]
        hma_f_prev = hma_fast[i-1]
        hma_s_prev = hma_slow[i-1]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: HMA(16) crosses above HMA(32), 1d uptrend, volume confirmation
            if hma_f > hma_s and hma_f_prev <= hma_s_prev and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: HMA(16) crosses below HMA(32), 1d downtrend, volume confirmation
            elif hma_f < hma_s and hma_f_prev >= hma_s_prev and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: HMA(16) crosses below HMA(32)
            if hma_f < hma_s and hma_f_prev >= hma_s_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: HMA(16) crosses above HMA(32)
            if hma_f > hma_s and hma_f_prev <= hma_s_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals