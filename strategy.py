#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_20_Volume_Trend
# Hypothesis: 4-hour Donchian channel breakout (20-period) with 1-day EMA trend filter and volume confirmation.
# Designed to work in both bull and bear markets by following the daily trend. Breakouts are filtered by volume
# to ensure strength, reducing false signals. Targets 20-50 trades per year to minimize fee drag.

name = "4h_Donchian_Breakout_20_20_Volume_Trend"
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
    
    # Daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Donchian channel (20-period) on 4h data
    def donchian_channels(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(window - 1, len(high)):
            upper[i] = np.max(high[i - window + 1:i + 1])
            lower[i] = np.min(low[i - window + 1:i + 1])
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: 20-period average on 4h data
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align daily indicators to 4h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, above daily EMA34, strong volume
            if close[i] > upper[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, below daily EMA34, strong volume
            elif close[i] < lower[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below lower Donchian or below daily EMA34
            if close[i] < lower[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper Donchian or above daily EMA34
            if close[i] > upper[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals