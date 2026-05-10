#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_VolumeConfirmation_1dTrend_Filter
# Hypothesis: Donchian(20) breakouts on 4h chart capture medium-term momentum. 
# Volume confirmation (volume > 1.5x 20-period average) filters false breakouts.
# Daily trend filter (close > EMA50) ensures alignment with higher timeframe trend.
# Designed for low trade frequency (~25-40/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (short breakdowns against trend).

name = "4h_Donchian_Breakout_20_VolumeConfirmation_1dTrend_Filter"
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
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Daily trend filter: EMA 50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get daily close for trend determination
        close_1d_series = pd.Series(close_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_series.values)
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian high + volume confirmation + daily uptrend
            if close[i] > high_roll[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low + volume confirmation + daily downtrend
            elif close[i] < low_roll[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low or daily trend turns down
            if close[i] < low_roll[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high or daily trend turns up
            if close[i] > high_roll[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals