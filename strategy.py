#!/usr/bin/env python3
"""
4h_RVOL_Spike_Donchian_Breakout_TrendFilter
Hypothesis: Donchian(20) breakout with volume spike confirmation (RVOL > 2.0) and EMA34 trend filter.
RVOL filters out low-volume false breakouts. Works in bull (breakouts continue) and bear (breakouts fail fast, stopped by trend filter).
Target: 20-35 trades/year to avoid fee drag.
"""

name = "4h_RVOL_Spike_Donchian_Breakout_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # RVOL: volume / 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) and EMA34 (34)
    start_idx = max(lookback, 34)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(rvol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with uptrend and volume spike
            if high[i] > highest_high[i] and close[i] > ema_34_1d_aligned[i] and rvol[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with downtrend and volume spike
            elif low[i] < lowest_low[i] and close[i] < ema_34_1d_aligned[i] and rvol[i] > 2.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price back below Donchian low or trend change
            if low[i] < lowest_low[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price back above Donchian high or trend change
            if high[i] > highest_high[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals