#!/usr/bin/env python3
"""
12h_1d_Donchian20_Breakout_1dTrend_VolumeFilter
Hypothesis: Donchian 20-period breakout on 12h timeframe, filtered by 1d trend (EMA34) and volume confirmation (>1.5x 20-period average), works in both bull and bear markets by following higher timeframe trend. Target: 12-37 trades/year per symbol.
"""

name = "12h_1d_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # Donchian channel (20-period) on 12h
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values for current bar
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        dh = donch_high[i]
        dl = donch_low[i]
        
        if position == 0:
            # LONG: price breaks above Donchian high, 1d uptrend, volume confirmation
            if close[i] > dh and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low, 1d downtrend, volume confirmation
            elif close[i] < dl and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low or 1d trend turns down
            if close[i] < dl or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high or 1d trend turns up
            if close[i] > dh or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals