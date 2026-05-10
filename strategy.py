#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_TrendFilter
Hypothesis: Combines Elder Ray Bull/Bear Power with 1d ADX trend filter to capture momentum in both trending and ranging markets. 
Bull Power measures buying strength (high - EMA13), Bear Power measures selling strength (low - EMA13). 
Long when Bull Power > 0 and Bear Power rising from negative, with 1d ADX > 25 (trending). 
Short when Bear Power < 0 and Bull Power falling from positive, with 1d ADX > 25. 
Uses 6m timeframe for entries with 1d trend filter to avoid counter-trend trades. 
Designed for low trade frequency (15-25/year) to minimize fee drag while capturing sustained moves.
"""

name = "6h_ElderRay_BullBearPower_TrendFilter"
timeframe = "6h"
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
    
    # Elder Ray: EMA13 of close
    ema_period = 13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=ema_period, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # 1d ADX for trend filter (need +DI, -DI, DX)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DM, -DM, TR
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(high_1d[1:] - close_1d[:-1], low_1d[1:] - close_1d[:-1])
    )
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            res[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                res[i] = res[i-1] - (res[i-1] / period) + arr[i]
        return res
    
    adx_period = 14
    tr_sum = wilder_smooth(tr, adx_period)
    dm_plus_sum = wilder_smooth(dm_plus, adx_period)
    dm_minus_sum = wilder_smooth(dm_minus, adx_period)
    
    # Avoid division by zero
    dx = np.where(tr_sum > 0, 
                  np.abs(dm_plus_sum - dm_minus_sum) / tr_sum * 100, 
                  0)
    
    # ADX is smoothed DX
    adx = wilder_smooth(dx, adx_period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_period, adx_period*2) + 5
    
    for i in range(start_idx, n):
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND Bear Power turning up from negative
            if bull_power[i] > 0 and bear_power[i] > bear_power[i-1] and bear_power[i-1] < 0 and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) AND Bull Power turning down from positive
            elif bear_power[i] < 0 and bull_power[i] < bull_power[i-1] and bull_power[i-1] > 0 and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR Bear Power turns positive (momentum fade)
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR Bull Power turns negative (momentum fade)
            if bear_power[i] >= 0 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals