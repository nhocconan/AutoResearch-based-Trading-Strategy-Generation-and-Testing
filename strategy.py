#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: 1h breakouts from Camarilla R1/S1 levels with 4h trend filter (EMA21) and 1d volume confirmation.
# Uses 4h for trend direction, 1d for volume filter, 1h for precise entry timing.
# Designed for 1h to achieve 15-37 trades/year with session filter (08-20 UTC) to reduce noise.

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
    
    # 4h data for trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 1d data for Camarilla levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 4h EMA21 for trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Camarilla levels (based on previous day) on 1d
    def calculate_camarilla(h, l, c):
        range_ = h - l
        R1 = c + (range_ * 1.0833 / 6)
        S1 = c - (range_ * 1.0833 / 6)
        return R1, S1
    
    R1 = np.full_like(close_1d, np.nan)
    S1 = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        R1[i], S1[i] = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Align indicators to 1h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_21_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Long: price breaks above R1, above 4h EMA21, strong 1d volume
            if close[i] > R1_aligned[i] and close[i] > ema_21_4h_aligned[i] and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, below 4h EMA21, strong 1d volume
            elif close[i] < S1_aligned[i] and close[i] < ema_21_4h_aligned[i] and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or below 4h EMA21
            if close[i] < S1_aligned[i] or close[i] < ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above R1 or above 4h EMA21
            if close[i] > R1_aligned[i] or close[i] > ema_21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals