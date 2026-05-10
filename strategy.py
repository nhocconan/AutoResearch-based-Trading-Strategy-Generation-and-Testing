#!/usr/bin/env python3
# 1D_KAMA_Direction_1wTrend_Filter_Volume
# Hypothesis: Use 1-day KAMA to determine trend direction, with 1-week trend filter (KAMA slope) and volume confirmation.
# This avoids overtrading by requiring alignment of short-term (1d) and long-term (1w) momentum, plus volume spike.
# Designed for low trade frequency (target: 10-20/year) with discrete position sizing (0.25) to minimize fee churn.
# Works in both bull and bear markets by requiring trend alignment across timeframes.

name = "1D_KAMA_Direction_1wTrend_Filter_Volume"
timeframe = "1d"
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
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend direction
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        noise = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(noise != 0, change / noise, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1d = kama(df_1d['close'].values, length=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1w KAMA slope for trend filter (rising/falling)
    kama_1w = kama(df_1w['close'].values, length=10, fast=2, slow=30)
    kama_1w_slope = np.diff(kama_1w, prepend=kama_1w[0])
    kama_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_slope)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_slope_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend: price above/below KAMA
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        
        # 1w trend: KAMA slope positive/negative
        kama_1w_rising = kama_1w_slope_aligned[i] > 0
        kama_1w_falling = kama_1w_slope_aligned[i] < 0
        
        if position == 0:
            # Long entry: price above 1d KAMA + 1w KAMA rising + volume spike
            if (price_above_kama and 
                kama_1w_rising and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 1d KAMA + 1w KAMA falling + volume spike
            elif (price_below_kama and 
                  kama_1w_falling and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d KAMA OR 1w KAMA turns down
            if (close[i] < kama_1d_aligned[i] or 
                kama_1w_slope_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d KAMA OR 1w KAMA turns up
            if (close[i] > kama_1d_aligned[i] or 
                kama_1w_slope_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals