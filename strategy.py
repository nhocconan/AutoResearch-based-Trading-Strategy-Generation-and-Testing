#!/usr/bin/env python3
# 1d_WeeklyPivot_Trend_Follow
# Hypothesis: Weekly pivot levels (S1, R1) act as strong support/resistance in trending markets.
# In bull markets, price bouncing off weekly S1 with upward momentum continues higher.
# In bear markets, price rejecting at weekly R1 with downward momentum continues lower.
# Uses 1-week trend filter (EMA50) to ensure trades align with higher-timeframe trend.
# Volume confirmation filters out false breakouts. Designed for low frequency (10-25 trades/year)
# to minimize fee drag and improve generalization across bull/bear cycles.

name = "1d_WeeklyPivot_Trend_Follow"
timeframe = "1d"
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
    
    # Get weekly data for trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = (2 * Pivot) - L
    # S1 = (2 * Pivot) - H
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_r1 = (2 * weekly_pivot) - df_1w['low']
    weekly_s1 = (2 * weekly_pivot) - df_1w['high']
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    
    # Volume confirmation (20-period MA on daily = ~20 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price crosses above weekly S1 + volume
            if uptrend and close[i] > weekly_s1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price crosses below weekly R1 + volume
            elif downtrend and close[i] < weekly_r1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price falls back below weekly S1
            if not uptrend or close[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price rises back above weekly R1
            if not downtrend or close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals