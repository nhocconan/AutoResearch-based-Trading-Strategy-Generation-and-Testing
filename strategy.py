#!/usr/bin/env python3
# 4h_1d_TrendFollow_VolumeBreakout
# Hypothesis: Price breaking above/below 1-day EMA34 with volume confirmation on 4h timeframe.
# Uses 1-day EMA34 for trend (smooth, captures multi-day bias) and volume spike to confirm breakout strength.
# Designed to work in both bull and bear markets by following established trends with volume confirmation.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

name = "4h_1d_TrendFollow_VolumeBreakout"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume average for spike detection
    vol_ma_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0 * 1d average volume
        volume_spike = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price > 1d EMA34 (uptrend) and breaks above with volume
            if close[i] > ema34_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < 1d EMA34 (downtrend) and breaks below with volume
            elif close[i] < ema34_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals