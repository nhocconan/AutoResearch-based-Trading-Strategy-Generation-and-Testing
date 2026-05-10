#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_TrendFilter_Volume
Hypothesis: Weekly pivot breakouts (R1/S1) on daily chart with 1w EMA trend filter and volume confirmation capture momentum in both bull and bear markets by trading breakouts of key weekly support/resistance levels. Timeframe: 1d balances trade frequency and signal quality for low turnover.
"""

name = "1d_WeeklyPivot_Breakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points using previous week's OHLC
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]  # First value uses current week's high as placeholder
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Calculate weekly pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    R1 = 2 * pivot - prev_low
    S1 = 2 * pivot - prev_high
    R2 = pivot + (prev_high - prev_low)
    S2 = pivot - (prev_high - prev_low)
    
    # Align weekly pivot levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Get 1w data for EMA trend filter (34-period EMA)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get 1d data for volume confirmation
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma20 * 1.5
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 (34 weeks)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or
            np.isnan(S2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1w EMA34
        uptrend_1w = close[i] > ema34_1w_aligned[i]
        downtrend_1w = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and uptrend
            if high[i] > R1_aligned[i] and uptrend_1w and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume and downtrend
            elif low[i] < S1_aligned[i] and downtrend_1w and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R2 or trend fails
            if high[i] >= R2_aligned[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S2 or trend fails
            if low[i] <= S2_aligned[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals