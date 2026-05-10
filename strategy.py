#!/usr/bin/env python3
# 6h_1w_Trend_1d_Pivot_Breakout_With_Volume
# Hypothesis: Use weekly trend (price above/below weekly EMA200) to set directional bias,
# then trade breakouts of daily pivot levels (R1/S1) with volume confirmation on 6h.
# Weekly trend provides multi-month bias to avoid whipsaws, while daily pivots offer
# precise entry/exit levels. Volume confirmation filters false breakouts.
# Works in bull markets by following weekly uptrend and buying dips to R1 breakouts.
# Works in bear markets by following weekly downtrend and selling rallies to S1 breakdowns.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_1w_Trend_1d_Pivot_Breakout_With_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    weekly_close = df_1w['close'].values
    ema_200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily pivot levels (standard formula)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot_point = (daily_high + daily_low + daily_close) / 3
    daily_r1 = 2 * pivot_point - daily_low
    daily_s1 = 2 * pivot_point - daily_high
    daily_pivot = pivot_point  # for exit condition
    
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Volume confirmation (24-period MA on 6h = ~6 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA200 (200) and volume MA (24)
    start_idx = max(200, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(daily_r1_aligned[i]) or 
            np.isnan(daily_s1_aligned[i]) or 
            np.isnan(daily_pivot_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation (>1.5x MA to balance sensitivity and filtering)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: weekly uptrend + price breaks above daily R1 + volume
            if uptrend and close[i] > daily_r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + price breaks below daily S1 + volume
            elif downtrend and close[i] < daily_s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR price breaks below daily pivot point
            if not uptrend or close[i] < daily_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR price breaks above daily pivot point
            if not downtrend or close[i] > daily_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals