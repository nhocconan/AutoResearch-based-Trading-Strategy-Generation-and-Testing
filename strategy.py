#!/usr/bin/env python3
# 4h_VolumeBreakout_EMA50_Trend
# Hypothesis: High-probability breakouts occur when price moves with strong volume beyond
# the 50-period EMA on the 4h chart. Volume > 1.5x average confirms institutional
# participation. The EMA50 acts as dynamic support/resistance and trend filter.
# Works in bull markets via breakouts above EMA50 and in bear via breakdowns below EMA50.
# Designed for low trade frequency (15-25/year) to minimize fee drag.

name = "4h_VolumeBreakout_EMA50_Trend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA50 on close for trend and dynamic support/resistance
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average (~10 hours)
    vol_ma_period = 20
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, vol_ma_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        if np.isnan(ema50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price crosses above EMA50 with volume confirmation
            if close[i] > ema50[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below EMA50 with volume confirmation
            elif close[i] < ema50[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below EMA50
            if close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above EMA50
            if close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

EOF