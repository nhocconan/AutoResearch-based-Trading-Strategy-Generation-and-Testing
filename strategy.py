#!/usr/bin/env python3
# 6H_Donchian_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: 6h Donchian(20) breakouts aligned with 1d EMA50 trend and volume > 2x 20-period average capture strong momentum moves. Works in bull/bear by following 1d trend direction. Targets 12-37 trades/year on 6f timeframe.

name = "6H_Donchian_Breakout_1dTrend_VolumeConfirmation"
timeframe = "6h"
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
    
    # 6h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above 6h Donchian upper + volume confirmation + 1d uptrend
            if (high[i] > high_20[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 6h Donchian lower + volume confirmation + 1d downtrend
            elif (low[i] < low_20[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below 6h Donchian lower
            if low[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above 6h Donchian upper
            if high[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals