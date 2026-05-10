#!/usr/bin/env python3
# 6h_Donchian_Breakout_WeeklyTrend_Volume
# Hypothesis: Price breaks above/below 6h Donchian(20) channels with volume confirmation and weekly trend filter.
# Weekly trend ensures alignment with higher timeframe momentum, reducing false breakouts in chop.
# Designed for low trade frequency (15-25/year) to minimize fee drift. Works in bull/bear via trend filter.

name = "6h_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend (smooth, lag-appropriate)
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h Donchian(20) channels
    def highest(arr, p):
        res = np.full_like(arr, np.nan)
        for i in range(p-1, len(arr)):
            res[i] = np.max(arr[i-p+1:i+1])
        return res
    def lowest(arr, p):
        res = np.full_like(arr, np.nan)
        for i in range(p-1, len(arr)):
            res[i] = np.min(arr[i-p+1:i+1])
        return res
    
    donchian_high = highest(high, 20)
    donchian_low = lowest(low, 20)
    
    # Volume confirmation (20-period average on 6h = ~5 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, above weekly EMA20 (uptrend)
            if close[i] > donchian_high[i] and volume_confirm and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, below weekly EMA20 (downtrend)
            elif close[i] < donchian_low[i] and volume_confirm and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low or breaks below weekly EMA20
            if close[i] < donchian_low[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high or breaks above weekly EMA20
            if close[i] > donchian_high[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals