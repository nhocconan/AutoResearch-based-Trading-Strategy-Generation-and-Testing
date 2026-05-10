#!/usr/bin/env python3
# 4H_Range_Breakout_VolumeFilter_1dTrend
# Hypothesis: Range breakouts with volume confirmation and daily trend filter capture momentum
# while avoiding false breakouts in choppy markets. Works in bull markets (breakouts with trend) 
# and bear markets (short breakdowns against trend). Uses daily EMA50 for trend filter to align
# with higher timeframe direction. Designed for low trade frequency (~20-40/year) with discrete
# sizing (0.25) to minimize fee drag.

name = "4H_Range_Breakout_VolumeFilter_1dTrend"
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
    
    # 20-period range (highest high, lowest low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Daily trend filter: EMA 50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily close aligned for trend determination
    close_1d_series = pd.Series(close_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_series.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above 20-period high + volume confirmation + daily uptrend
            if close[i] > highest_high[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 20-period low + volume confirmation + daily downtrend
            elif close[i] < lowest_low[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below 20-period low or daily trend turns down
            if close[i] < lowest_low[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above 20-period high or daily trend turns up
            if close[i] > highest_high[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals