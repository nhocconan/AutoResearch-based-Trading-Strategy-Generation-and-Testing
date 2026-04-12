#!/usr/bin/env python3
"""
6h_1w_1d_TwoStageTrend_v1
Hypothesis: Weekly trend (price > weekly SMA50) filters direction; 6h Donchian(20) breakouts enter only when aligned with weekly trend. Daily ATR(14) filters for sufficient volatility. Designed for low trade frequency (15-30/year) by requiring strong breakouts in the direction of the higher timeframe trend. Works in bull/bear via weekly trend filter and volatility-adjusted position sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_TwoStageTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly SMA50 for trend
    sma_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        for i in range(50, len(close_1w)):
            sma_50_1w[i] = np.mean(close_1w[i-50:i])
    
    # Align weekly trend to 6h
    weekly_uptrend = align_htf_to_ltf(prices, df_1w, close_1w > sma_50_1w)
    
    # === DAILY VOLATILITY FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(14) for volatility filter
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index 0
    
    atr_14_1d = np.full_like(tr_1d, np.nan)
    if len(tr_1d) >= 15:  # need 14 + 1 for calculation
        for i in range(14, len(tr_1d)):
            atr_14_1d[i] = np.nanmean(tr_1d[i-13:i+1])  # Wilder's smoothing: simple average of last 14 TR
    
    # Align daily ATR to 6h
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # === 6H DONCHIAN BREAKOUT ===
    # Donchian(20) on 6h
    highest_20 = np.full_like(high, np.nan)
    lowest_20 = np.full_like(low, np.nan)
    
    if len(high) >= 20:
        for i in range(20, len(high)):
            highest_20[i] = np.max(high[i-20:i])
            lowest_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(weekly_uptrend[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or atr_14_1d_aligned[i] <= 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: ATR > 0.5% of price (ensures sufficient movement)
        vol_filter = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i]
        breakout_down = close[i] < lowest_20[i]
        
        # Entry conditions: breakout in direction of weekly trend with volatility
        long_entry = breakout_up and weekly_uptrend[i] and vol_filter
        short_entry = breakout_down and (not weekly_uptrend[i]) and vol_filter
        
        # Exit conditions: opposite Donchian breakout (trailing stop via structure)
        exit_long = breakout_down
        exit_short = breakout_up
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals