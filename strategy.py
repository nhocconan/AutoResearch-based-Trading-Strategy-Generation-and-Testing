#!/usr/bin/env python3
name = "4h_Donchian20_VolumeTrend_12hTrendFilter"
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
    
    # Get 12H data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # EMA50 on 12H for trend
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian(20) on 4H
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(n):
        if i < 20:
            donch_high[i] = np.max(high[:i+1])
            donch_low[i] = np.min(low[:i+1])
        else:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above 12H EMA50 + volume surge
            if close[i] > donch_high[i] and close[i] > ema50_12h_aligned[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below 12H EMA50 + volume surge
            elif close[i] < donch_low[i] and close[i] < ema50_12h_aligned[i] and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR below 12H EMA50
            if close[i] < donch_low[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR above 12H EMA50
            if close[i] > donch_high[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals