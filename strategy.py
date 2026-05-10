#!/usr/bin/env python3
"""
4h_4WMA_Cross_1dTrend_Volume
Hypothesis: 4-period and 16-period WMA crossover on 4h with 1d EMA34 trend filter and volume confirmation.
WMA (Weighted Moving Average) gives more weight to recent prices, making it more responsive than EMA.
Crossover signals trend changes. Trend filter ensures alignment with higher timeframe direction.
Volume confirmation filters weak signals. Works in both bull (buy on bullish cross) and bear (sell on bearish cross).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "4h_4WMA_Cross_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # 4-period and 16-period WMA on 4h
    wma4 = np.full(n, np.nan)
    wma16 = np.full(n, np.nan)
    
    if n >= 4:
        # WMA(4): weights [1,2,3,4] sum=10
        weights4 = np.array([1, 2, 3, 4], dtype=float)
        sum_weights4 = weights4.sum()
        for i in range(3, n):
            wma4[i] = np.dot(close[i-3:i+1], weights4) / sum_weights4
    
    if n >= 16:
        # WMA(16): weights [1,2,...,16] sum=136
        weights16 = np.arange(1, 17, dtype=float)
        sum_weights16 = weights16.sum()
        for i in range(15, n):
            wma16[i] = np.dot(close[i-15:i+1], weights16) / sum_weights16
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 16)  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(wma4[i]) or np.isnan(wma16[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: WMA4 crosses above WMA16 with uptrend and volume confirmation
            if wma4[i] > wma16[i] and wma4[i-1] <= wma16[i-1] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: WMA4 crosses below WMA16 with downtrend and volume confirmation
            elif wma4[i] < wma16[i] and wma4[i-1] >= wma16[i-1] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: WMA4 crosses below WMA16 or trend reversal
            if wma4[i] < wma16[i] and wma4[i-1] >= wma16[i-1] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: WMA4 crosses above WMA16 or trend reversal
            if wma4[i] > wma16[i] and wma4[i-1] <= wma16[i-1] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals