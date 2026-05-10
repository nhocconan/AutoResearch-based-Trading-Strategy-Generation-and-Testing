#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels from daily timeframe act as significant support/resistance.
Breakouts above R3 or below S3 with weekly trend confirmation and volume spikes capture
strong momentum moves. Weekly trend filter reduces whipsaws in choppy markets.
Designed for low trade frequency (15-30/year) to minimize fee drag while capturing
trend moves in both bull and bear markets.
"""

name = "12h_Camarilla_Pivot_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Weekly trend filter - EMA34 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    # Pivot = (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # Range = H-L
    range_1d = high_1d - low_1d
    # R3 = Pivot + (Range * 1.1)
    r3_1d = pivot_1d + (range_1d * 1.1)
    # S3 = Pivot - (Range * 1.1)
    s3_1d = pivot_1d - (range_1d * 1.1)
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume confirmation (24-period average = 12 days on 12h timeframe)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough history for weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or \
           np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend: price above/below EMA34
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R3, weekly uptrend, volume confirmation
            if close[i] > r3_1d_aligned[i] and weekly_uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, weekly downtrend, volume confirmation
            elif close[i] < s3_1d_aligned[i] and weekly_downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops back below pivot OR weekly trend turns down
            if close[i] < pivot_1d[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above pivot OR weekly trend turns up
            if close[i] > pivot_1d[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals