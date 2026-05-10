#!/usr/bin/env python3
# 4H_Donchian_Breakout_20_VolumeConfirmation_1dTrend_Filter
# Hypothesis: Breakout of 20-period Donchian channel with volume confirmation and daily EMA50 trend filter.
# Uses 4h timeframe to balance trade frequency and signal quality. Long when price breaks above upper band
# with volume > 1.5x 20-period average and price > daily EMA50 (uptrend). Short when price breaks below
# lower band with volume confirmation and price < daily EMA50 (downtrend). Exits on opposite band touch.
# Designed for 15-30 trades/year to avoid fee drag while capturing trend moves.

name = "4H_Donchian_Breakout_20_VolumeConfirmation_1dTrend_Filter"
timeframe = "4h"
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
    
    # 4h Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    upper_band = rolling_max(high, 20)
    lower_band = rolling_min(low, 20)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period volume average for confirmation
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align 1d EMA50 to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above upper band with volume and uptrend
            if close[i] > upper_band[i] and volume_condition and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band with volume and downtrend
            elif close[i] < lower_band[i] and volume_condition and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches or crosses lower band
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches or crosses upper band
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals