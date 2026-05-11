#!/usr/bin/env python3
"""
6h_ADX_Slope_Trend_12hEMA34_Filter
Hypothesis: Use ADX slope (trend acceleration) on 6h as primary signal, filtered by 12h EMA34 direction. Works in bull/bear by capturing momentum shifts. Low-frequency entries avoid fee drag.
"""

name = "6h_ADX_Slope_Trend_12hEMA34_Filter"
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
    
    # Calculate ADX (14) on 6h
    period = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period-1] = np.nansum(tr[:period]) if period <= n else 0
    plus_dm_sum = np.nansum(plus_dm[:period]) if period <= n else 0
    minus_dm_sum = np.nansum(minus_dm[:period]) if period <= n else 0
    
    if atr[period-1] > 0:
        plus_di[period-1] = 100 * plus_dm_sum / atr[period-1]
        minus_di[period-1] = 100 * minus_dm_sum / atr[period-1]
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / period * 100 / atr[i] if atr[i] > 0 else 0
        minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / period * 100 / atr[i] if atr[i] > 0 else 0
    
    dx = np.zeros(n)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros(n)
    if period*2-1 < n:
        adx[period*2-1] = np.nanmean(dx[period:period*2]) if np.sum(~np.isnan(dx[period:period*2])) > 0 else 0
        for i in range(period*2, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # ADX slope (rate of change over 3 periods)
    adx_slope = np.zeros(n)
    for i in range(3, n):
        if not np.isnan(adx[i]) and not np.isnan(adx[i-3]):
            adx_slope[i] = (adx[i] - adx[i-3]) / 3
    
    # Get 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Signals
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup
    start_idx = max(period*2, 34) + 3
    
    for i in range(start_idx, n):
        # Skip if any data invalid
        if (np.isnan(adx_slope[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(close[i]) or
            np.isnan(high[i]) or
            np.isnan(low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX rising (momentum building) AND price above 12h EMA34 (uptrend)
            if adx_slope[i] > 0.15 and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX rising AND price below 12h EMA34 (downtrend)
            elif adx_slope[i] > 0.15 and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX slope turns negative (momentum fading) OR price crosses below EMA
            if adx_slope[i] < -0.05 or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX slope turns negative OR price crosses above EMA
            if adx_slope[i] < -0.05 or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals