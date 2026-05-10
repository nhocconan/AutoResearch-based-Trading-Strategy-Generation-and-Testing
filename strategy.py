#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: 1-hour breakouts from 4-hour Camarilla R1/S1 levels with 4-hour trend filter (EMA34) and volume confirmation.
# Uses 4h for signal direction (trend and levels) and 1h for entry timing to reduce false breakouts.
# Designed for 1h to achieve 15-37 trades/year, suitable for both bull and bear markets.
# Session filter (08-20 UTC) reduces noise trades.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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
    
    # 4-hour data for EMA34 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4-hour EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 4-hour Camarilla levels (based on previous 4h bar)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R1 = c + (range_ * 1.1000 / 12)
        S1 = c - (range_ * 1.1000 / 12)
        return R1, S1
    
    R1 = np.full_like(close_4h, np.nan)
    S1 = np.full_like(close_4h, np.nan)
    for i in range(1, len(close_4h)):
        R1[i], S1[i] = calculate_camarilla(high_4h[i-1], low_4h[i-1], close_4h[i-1])
    
    # 4-hour volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_4h = mean_arr(volume_4h, 20)
    
    # Align 4-hour indicators to 1h timeframe (wait for 4h bar to close)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above 4h EMA34, strong volume
            if close[i] > R1_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > 2.0 * vol_ma_20_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, below 4h EMA34, strong volume
            elif close[i] < S1_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > 2.0 * vol_ma_20_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or below 4h EMA34
            if close[i] < S1_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above R1 or above 4h EMA34
            if close[i] > R1_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals