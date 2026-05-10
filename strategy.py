#!/usr/bin/env python3
# 6H_Donchian_Breakout_20_WeeklyPivotDirection_VolumeConfirmation
# Hypothesis: Donchian(20) breakouts on 6h chart capture medium-term momentum.
# Weekly pivot direction (from 1w timeframe) provides higher-trend filter.
# Volume confirmation (volume > 1.5x 20-period average) filters false breakouts.
# Designed for low trade frequency (~15-25/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (short breakdowns against trend).

name = "6H_Donchian_Breakout_20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Weekly trend filter: EMA 50 from 1w data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_uptrend = close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian high + volume confirmation + weekly uptrend
            if close[i] > high_roll[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low + volume confirmation + weekly downtrend
            elif close[i] < low_roll[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low or weekly trend turns down
            if close[i] < low_roll[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high or weekly trend turns up
            if close[i] > high_roll[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals