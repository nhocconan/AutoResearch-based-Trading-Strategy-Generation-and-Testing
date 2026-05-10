#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Weekly Camarilla R1/S1 levels act as significant support/resistance.
# Breakout above weekly R1 with 1-week EMA34 trend filter and volume confirmation
# captures sustained moves. Works in bull markets (breakouts above R1) and bear markets
# (breakdowns below S1) by following the 1-week trend. Low trade frequency expected due
# to strict breakout conditions + trend filter + volume confirmation.

name = "1d_Weekly_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from previous weekly bar
    # R1 = close + 1.1 * (high - low) / 4
    # S1 = close - 1.1 * (high - low) / 4
    cam_range = high_1w - low_1w
    r1 = close_1w + 1.1 * cam_range / 4
    s1 = close_1w - 1.1 * cam_range / 4
    
    # Align Camarilla levels to daily timeframe (wait for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (20-period average ~ 20 days for daily timeframe)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough history for EMA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above weekly EMA34, volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema_34_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below weekly EMA34, volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema_34_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or below weekly EMA34
            if close[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 or above weekly EMA34
            if close[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals