#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# Breakouts above R3 or below S3 with 1d trend confirmation (price vs EMA34) and volume spike
# lead to continuation moves. Works in both bull and bear markets as it captures breakouts
# from key daily levels. Designed for low trade frequency (12-37/year) to minimize fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    close_1d_shift = close_1d  # same day close for same-day levels
    range_1d = high_1d - low_1d
    r3 = close_1d_shift + 1.1 * range_1d
    s3 = close_1d_shift - 1.1 * range_1d
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily values to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (24-period average on 12h = ~12 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirm = volume[i] > 2.0 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R3 with volume, above 1d EMA34 (uptrend)
            if close[i] > r3_aligned[i] and volume_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, below 1d EMA34 (downtrend)
            elif close[i] < s3_aligned[i] and volume_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 or breaks below 1d EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 or breaks above 1d EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals