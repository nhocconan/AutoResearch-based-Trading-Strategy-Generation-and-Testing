#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: 1-hour breakouts from 4-hour Camarilla R1/S1 levels with 4-hour trend filter (EMA21) and 1-day volume confirmation.
# Uses 4h for signal direction (trend + Camarilla levels) and 1h only for entry timing precision.
# 1d volume filter ensures breakouts occur with institutional participation, reducing false signals.
# Designed for 16-35 trades/year on 1h timeframe to avoid fee drag while capturing true breakouts.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
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
    
    # 4-hour data for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4-hour EMA21 for trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4-hour Camarilla levels (based on previous 4h bar)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R1 = c + (range_ * 1.1000 / 12)
        S1 = c - (range_ * 1.1000 / 12)
        return R1, S1
    
    R1_4h = np.full_like(close_4h, np.nan)
    S1_4h = np.full_like(close_4h, np.nan)
    for i in range(1, len(close_4h)):
        R1_4h[i], S1_4h[i] = calculate_camarilla(high_4h[i-1], low_4h[i-1], close_4h[i-1])
    
    # 1-day volume confirmation: 20-period average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align 4h indicators to 1h timeframe (wait for 4h bar to close)
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # Align 1d volume to 1h timeframe (wait for 1d bar to close)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or \
           np.isnan(ema_21_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1_4h, above 4h EMA21, strong 1d volume
            if close[i] > R1_4h_aligned[i] and close[i] > ema_21_4h_aligned[i] and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1_4h, below 4h EMA21, strong 1d volume
            elif close[i] < S1_4h_aligned[i] and close[i] < ema_21_4h_aligned[i] and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price drops below S1_4h or below 4h EMA21
            if close[i] < S1_4h_aligned[i] or close[i] < ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above R1_4h or above 4h EMA21
            if close[i] > R1_4h_aligned[i] or close[i] > ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals