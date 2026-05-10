#!/usr/bin/env python3
# 1h_MomentumBreakout_4hTrend_1dVolume
# Hypothesis: 1-hour momentum breakouts filtered by 4-hour trend direction and 1-day volume confirmation.
# The 4-hour EMA50 determines trend (bullish if price > EMA50, bearish if price < EMA50).
# 1-hour breakouts occur when price crosses the 20-period high/low with volume > 1.5x 20-period average.
# In bullish 4h trend, only long breakouts are taken; in bearish 4h trend, only short breakdowns.
# 1-day volume filter ensures sufficient market participation. Designed for 1h to achieve 15-37 trades/year.

name = "1h_MomentumBreakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d data for volume confirmation (20-period average)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # 1h indicators: 20-period high/low for breakout
    def highest(arr, p):
        res = np.full_like(arr, np.nan)
        for i in range(p - 1, len(arr)):
            res[i] = np.max(arr[i - p + 1:i + 1])
        return res
    def lowest(arr, p):
        res = np.full_like(arr, np.nan)
        for i in range(p - 1, len(arr)):
            res[i] = np.min(arr[i - p + 1:i + 1])
        return res
    high_20 = highest(high, 20)
    low_20 = lowest(low, 20)
    
    # Align 4h trend to 1h (wait for 4h bar to close)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Align 1d volume MA to 1h (wait for 1d bar to close)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or \
           np.isnan(high_20[i]) or np.isnan(low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend: bullish if close > EMA50, bearish if close < EMA50
        # Use current 4h close price (need to align 4h close to 1h)
        close_4h_series = pd.Series(close_4h)
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h_series.values)
        is_bullish_trend = close_4h_aligned[i] > ema_50_4h_aligned[i]
        is_bearish_trend = close_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Volume condition: current 1h volume > 1.5x 20-day average volume
        volume_condition = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price crosses above 20-period high, bullish 4h trend, volume confirmation
            if close[i] > high_20[i] and is_bullish_trend and volume_condition:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price crosses below 20-period low, bearish 4h trend, volume confirmation
            elif close[i] < low_20[i] and is_bearish_trend and volume_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below 20-period low or 4h trend turns bearish
            if close[i] < low_20[i] or not is_bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above 20-period high or 4h trend turns bullish
            if close[i] > high_20[i] or not is_bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals