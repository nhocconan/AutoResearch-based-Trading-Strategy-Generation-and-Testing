#!/usr/bin/env python3
# 4h_Donchian_Breakout_20_Trend_1dEMA50_Volume
# Hypothesis: 4-hour breakouts from Donchian channel (20) with daily EMA50 trend filter and volume confirmation.
# Daily EMA50 filters trend direction to avoid counter-trend trades; Donchian breakout provides clear entry;
# Volume confirmation ensures breakout strength. Designed for 4h to achieve 20-50 trades/year, suitable for both bull and bear markets.

name = "4h_Donchian_Breakout_20_Trend_1dEMA50_Volume"
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
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian channel (20-period) on 4h data
    def donchian_channel(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(window - 1, len(high)):
            upper[i] = np.max(high[i - window + 1:i + 1])
            lower[i] = np.min(low[i - window + 1:i + 1])
        return upper, lower
    
    upper, lower = donchian_channel(high, low, 20)
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 4h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper, above daily EMA50, strong volume
            if close[i] > upper[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, below daily EMA50, strong volume
            elif close[i] < lower[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below Donchian lower or below daily EMA50
            if close[i] < lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Donchian upper or above daily EMA50
            if close[i] > upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals