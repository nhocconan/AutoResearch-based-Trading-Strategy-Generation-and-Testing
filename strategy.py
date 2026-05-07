#!/usr/bin/env python3
# 6h_Supertrend_1wTrend_1dVolume
# Hypothesis: Combines Supertrend (ATR-based trend filter) on 6h with 1w trend direction and 1d volume spike to capture strong trends while avoiding whipsaws. The 1w trend ensures we only trade in the direction of the higher timeframe momentum, reducing false signals in sideways markets. Volume spike confirms institutional participation. Designed for 6h timeframe with target of 15-30 trades/year to minimize fee drag.

name = "6h_Supertrend_1wTrend_1dVolume"
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
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close)
    atr[atr_period] = np.mean(tr[:atr_period])
    for i in range(atr_period + 1, len(atr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Supertrend
    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close)
    dir = np.ones_like(close, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = lowerband[0]
    dir[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upperband[i-1]:
            dir[i] = 1
        elif close[i] < lowerband[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
            if dir[i] == -1 and lowerband[i] > lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if dir[i] == 1 and upperband[i] < upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if dir[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Daily volume moving average (20-day)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 34, 20)  # Ensure we have all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(supertrend[i]) or np.isnan(dir[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or vol_ma_1d_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend, weekly uptrend, and volume spike
            if dir[i] == 1 and close[i] > ema_34_1w_aligned[i] and volume[i] > 2.0 * vol_ma_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend, weekly downtrend, and volume spike
            elif dir[i] == -1 and close[i] < ema_34_1w_aligned[i] and volume[i] > 2.0 * vol_ma_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Supertrend turns down
            if dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Supertrend turns up
            if dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals