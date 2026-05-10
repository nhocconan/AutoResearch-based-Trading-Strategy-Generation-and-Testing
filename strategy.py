#!/usr/bin/env python3
"""
4h_RSI_1D_Trend_Filter
Hypothesis: In strong 1d trends (EMA150), 4h RSI(14) pullbacks offer high-probability entries with favorable risk-reward.
Works in bull markets via buying dips in uptrends and in bear markets via selling rallies in downtrends.
Uses 1d EMA150 for trend and 4h RSI(14) for entry timing, with volume confirmation to avoid false signals.
Designed for low trade frequency (target: 15-30 trades per year) to minimize fee drag.
"""

name = "4h_RSI_1D_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA150)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 150:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA150 for trend
    ema150_1d = pd.Series(close_1d).ewm(span=150, adjust=False, min_periods=150).mean().values
    ema150_1d_aligned = align_htf_to_ltf(prices, df_1d, ema150_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h volume average (20-period EMA) for volume filter
    vol_avg = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) and 1d EMA150 (150)
    start_idx = max(14, 150)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema150_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend_1d = close[i] > ema150_1d_aligned[i]
        downtrend_1d = close[i] < ema150_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 4h average volume
        volume_filter = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long entry: RSI < 40 (pullback) + uptrend + volume
            if rsi[i] < 40 and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI > 60 (pullback) + downtrend + volume
            elif rsi[i] > 60 and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 60 or trend fails
            if rsi[i] > 60 or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 40 or trend fails
            if rsi[i] < 40 or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals