#!/usr/bin/env python3
# 6H_WeeklyPivot_DailyTrend_Volume
# Hypothesis: Trade breakouts from weekly pivot levels (R4/S4) in direction of daily trend with volume confirmation.
# Weekly pivot levels act as strong support/resistance. Breakouts indicate momentum.
# Daily trend filter ensures we trade with higher timeframe momentum.
# Volume confirmation filters false breakouts.
# Works in bull/bear by following daily trend and using weekly structure for entries.
# Target: 15-30 trades/year per symbol.

name = "6H_WeeklyPivot_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot levels from 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We'll use the most recent completed week's data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support and resistance levels
    # R4 = Close + 3*(High - Low)  (aggressive breakout level)
    # S4 = Close - 3*(High - Low)
    weekly_range = weekly_high - weekly_low
    weekly_r4 = weekly_close + 3.0 * weekly_range
    weekly_s4 = weekly_close - 3.0 * weekly_range
    
    # Align weekly levels to 6h (weekly levels change only when new weekly bar starts)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.8  # Require strong volume for breakout
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price breaks above weekly R4 + daily uptrend + volume confirmation
            if daily_up and volume_confirm and close[i] > weekly_r4_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S4 + daily downtrend + volume confirmation
            elif daily_down and volume_confirm and close[i] < weekly_s4_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly pivot or trend changes
            if not daily_up or close[i] < weekly_pivot[len(weekly_pivot)-1] if len(weekly_pivot) > 0 else weekly_pivot[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot or trend changes
            if not daily_down or close[i] > weekly_pivot[len(weekly_pivot)-1] if len(weekly_pivot) > 0 else weekly_pivot[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals