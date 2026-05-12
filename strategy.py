#!/usr/bin/env python3
"""
6H_LR_CROSSOVER_1D_TREND_FILTER
Hypothesis: Linear regression crossover on 6h with 1-day trend filter captures medium-term momentum with reduced whipsaw.
Linear regression slope acts as adaptive trend strength filter - only take trades when 6h momentum aligns with 1-day trend.
Designed for 15-30 trades/year on 6h to minimize fee drag while capturing sustained moves in both bull and bear markets.
"""
name = "6H_LR_CROSSOVER_1D_TREND_FILTER"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 60-period linear regression slope on 6h (equivalent to ~15 days)
    def linreg_slope(arr, window):
        slopes = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window-1, len(arr)):
            y = arr[i-window+1:i+1]
            x = np.arange(len(y))
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes[i] = slope
        return slopes
    
    lr_slope_60 = linreg_slope(close, 60)
    
    # 1-day trend filter: 50-period EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after LR warmup
        if (np.isnan(lr_slope_60[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Positive 6h LR slope AND price above 1-day EMA50 (bullish alignment)
            if lr_slope_60[i] > 0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Negative 6h LR slope AND price below 1-day EMA50 (bearish alignment)
            elif lr_slope_60[i] < 0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 6h LR slope turns negative OR price crosses below 1-day EMA50
            if lr_slope_60[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 6h LR slope turns positive OR price crosses above 1-day EMA50
            if lr_slope_60[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals