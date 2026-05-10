#!/usr/bin/env python3
# 4h_Donchian_Breakout_12hTrend_Volume
# Hypothesis: 4-hour breakouts from Donchian(20) channels with 12-hour EMA50 trend filter and volume confirmation.
# The Donchian channel captures breakout momentum, while the 12-hour EMA50 ensures trend alignment
# to avoid counter-trend trades. Volume confirmation filters weak breakouts. Designed for 4h
# to achieve 20-50 trades/year, suitable for both bull and bear markets.

name = "4h_Donchian_Breakout_12hTrend_Volume"
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
    
    # 12-hour data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12-hour EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Donchian(20) channel
    def donchian_channel(high, low, lookback=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(lookback - 1, len(high)):
            upper[i] = np.max(high[i - lookback + 1:i + 1])
            lower[i] = np.min(low[i - lookback + 1:i + 1])
        return upper, lower
    
    upper, lower = donchian_channel(high, low, 20)
    
    # 12-hour volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    
    vol_ma_20_12h = mean_arr(volume_12h, 20)
    
    # Align 12-hour indicators to 4h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, above 12h EMA50, strong volume
            if close[i] > upper[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, below 12h EMA50, strong volume
            elif close[i] < lower[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below lower Donchian or below 12h EMA50
            if close[i] < lower[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper Donchian or above 12h EMA50
            if close[i] > upper[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals