#!/usr/bin/env python3
"""
4h_Bollinger_Band_Momentum_1dTrend_Volume
Hypothesis: Bollinger Band breakout with 1d EMA50 trend filter and volume confirmation.
Works in both bull and bear markets by following 1d trend and using volatility-based entries.
Target: 20-30 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "4h_Bollinger_Band_Momentum_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2)
    basis = np.full(n, np.nan)
    dev = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        basis[i] = np.mean(close[i-20:i])
        dev[i] = np.std(close[i-20:i])
        upper[i] = basis[i] + 2 * dev[i]
        lower[i] = basis[i] - 2 * dev[i]
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume SMA(20)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        if np.isnan(basis[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = volume[i] > 1.8 * vol_sma[i]
        
        if position == 0:
            # Long: Break above upper band with uptrend and volume confirmation
            if close[i] > upper[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band with downtrend and volume confirmation
            elif close[i] < lower[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses back below basis
            if close[i] < basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses back above basis
            if close[i] > basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals