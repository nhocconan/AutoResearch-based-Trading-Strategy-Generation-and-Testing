#!/usr/bin/env python3
name = "6h_WeeklyPivot_TrendBreak_1dVolatilityFilter"
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
    volume = prices['volume'].values
    
    # Weekly pivot points (calculated from previous week)
    df_1w = get_htf_data(prices, '1w')
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    # Weekly trend: close above/below pivot
    weekly_trend_up = close_w > pivot_w
    weekly_trend_down = close_w < pivot_w
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Daily volatility filter: ATR(14) normalized by price
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_norm = atr14 / close_1d
    atr_norm_aligned = align_htf_to_ltf(prices, df_1d, atr_norm)
    
    # 6h price breakout above/below weekly pivot levels
    breakout_up = close > r1_w[-1] if len(r1_w) > 0 else False  # placeholder, will be replaced in loop
    breakout_down = close < s1_w[-1] if len(s1_w) > 0 else False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # ensure ATR has enough data
    
    for i in range(start_idx, n):
        # Get weekly pivot levels for this point (using aligned arrays)
        # We need to get the values from the weekly arrays for the corresponding time
        # Since we aligned the weekly trend, we need to get the actual pivot values
        # We'll compute them on the fly from aligned weekly data
        
        # Get current weekly pivot levels by indexing into the weekly arrays
        # We need to find which week this 6h bar belongs to
        # Instead, we'll use the aligned weekly trend and compute levels from weekly data
        
        # For simplicity, we'll use the last available weekly pivot levels
        # In practice, we should align the pivot levels themselves
        # Let's align the pivot and S1/R1 levels
        
        # Recompute: align the actual pivot levels
        pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
        r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
        s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
        
        # Skip if data not ready
        if np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or np.isnan(atr_norm_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is above median (avoid choppy markets)
        vol_median = np.nanmedian(atr_norm_aligned[:i+1])
        vol_filter = atr_norm_aligned[i] > vol_median if not np.isnan(vol_median) else True
        
        if position == 0:
            # Long: weekly trend up + price breaks above R1 + volatility filter
            if (weekly_trend_up_aligned[i] and 
                close[i] > r1_w_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: weekly trend down + price breaks below S1 + volatility filter
            elif (weekly_trend_down_aligned[i] and 
                  close[i] < s1_w_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly pivot OR weekly trend changes
            if close[i] < pivot_w_aligned[i] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly pivot OR weekly trend changes
            if close[i] > pivot_w_aligned[i] or not weekly_trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals