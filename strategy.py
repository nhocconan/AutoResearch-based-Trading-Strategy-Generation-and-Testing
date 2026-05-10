#!/usr/bin/env python3
# 12H_Camarilla_Pivot_S1R1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (S1/R1) on 12h chart act as strong support/resistance. 
# Breakout above R1 with volume > 1.5x 20-period average and daily uptrend triggers long.
# Breakdown below S1 with volume > 1.5x 20-period average and daily downtrend triggers short.
# Exit when price returns to the pivot point (P) or trend reverses.
# Uses daily trend filter and volume confirmation to avoid false breakouts.
# Designed for low trade frequency (~15-30/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend).

name = "12H_Camarilla_Pivot_S1R1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Using daily data to calculate pivots for the current 12h bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    s1 = prev_close - (range_ * 1.1 / 12)
    r1 = prev_close + (range_ * 1.1 / 12)
    s2 = prev_close - (range_ * 1.1 / 6)
    r2 = prev_close + (range_ * 1.1 / 6)
    
    # Align daily pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Daily trend filter: EMA 50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(vol_threshold[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        is_uptrend = close_1d[i-1] > ema_50_1d[i-1] if i > 0 else False
        is_downtrend = close_1d[i-1] < ema_50_1d[i-1] if i > 0 else False
        
        if position == 0:
            # Long entry: Price breaks above R1 with volume confirmation and daily uptrend
            if close[i] > r1_aligned[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1 with volume confirmation and daily downtrend
            elif close[i] < s1_aligned[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to pivot point or daily trend turns down
            if close[i] <= pivot_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to pivot point or daily trend turns up
            if close[i] >= pivot_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals