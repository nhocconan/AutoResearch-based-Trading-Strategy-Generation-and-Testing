#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_12hTrend_Volume
# Hypothesis: Breakouts from weekly pivot levels on 6h with 12h trend filter (EMA34) and volume confirmation.
# Weekly pivots provide long-term institutional support/resistance; EMA34 on 12h filters trend direction; volume confirms breakout strength.
# Designed for 6h to achieve 12-37 trades/year, suitable for both bull and bear markets.

name = "6h_Weekly_Pivot_Breakout_12hTrend_Volume"
timeframe = "6h"
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
    
    # 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 1d data for weekly pivot calculation (need daily data to compute weekly)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly Pivot calculation (based on previous week)
    # We need to compute weekly OHLC from daily data
    # For each 6h bar, we need the weekly pivot from the previous completed week
    
    # First, compute daily typical price and use it to estimate weekly pivot
    # More accurate: group daily data into weeks, but we'll approximate using rolling window
    # Weekly high = max of high over past 5 trading days
    # Weekly low = min of low over past 5 trading days  
    # Weekly close = close of most recent day
    
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
    
    # Weekly high/low/close from daily data (5 trading days per week)
    weekly_high = rolling_max(high_1d, 5)
    weekly_low = rolling_min(low_1d, 5)
    weekly_close = np.roll(close_1d, 1)  # Previous day's close as weekly close approximation
    weekly_close[0] = np.nan
    
    # Weekly pivot levels (standard calculation)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    R1 = 2 * weekly_pivot - weekly_low
    S1 = 2 * weekly_pivot - weekly_high
    R2 = weekly_pivot + weekly_range
    S2 = weekly_pivot - weekly_range
    R3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    S3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Use R3/S3 as breakout levels (more significant than R1/S1)
    weekly_R3 = R3
    weekly_S3 = S3
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average on 12h
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_12h = mean_arr(volume_12h, 20)
    
    # Align all indicators to 6h timeframe (wait for 12h/1d bar to close)
    weekly_R3_aligned = align_htf_to_ltf(prices, df_1d, weekly_R3)
    weekly_S3_aligned = align_htf_to_ltf(prices, df_1d, weekly_S3)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(weekly_R3_aligned[i]) or np.isnan(weekly_S3_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3, above 12h EMA34, strong volume
            if close[i] > weekly_R3_aligned[i] and close[i] > ema_34_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, below 12h EMA34, strong volume
            elif close[i] < weekly_S3_aligned[i] and close[i] < ema_34_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below weekly S3 or below 12h EMA34
            if close[i] < weekly_S3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above weekly R3 or above 12h EMA34
            if close[i] > weekly_R3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals