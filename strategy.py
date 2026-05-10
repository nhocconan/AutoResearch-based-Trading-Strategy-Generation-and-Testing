#!/usr/bin/env python3
# 12h_Pivot_D1_Trend_Reversal
# Hypothesis: Reversal trades at daily pivot levels with 1d trend filter and volume confirmation on 12h timeframe.
# Uses daily pivot points (support/resistance) for mean reversion in ranging markets and trend alignment for filtering.
# Target: 15-35 trades/year to minimize fee drag on 12h timeframe, works in bull/bear via trend alignment.

name = "12h_Pivot_D1_Trend_Reversal"
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
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma[i] = vol_sum / 24
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Calculate daily pivot points from previous day
    df_1d_prev = get_htf_data(prices, '1d')
    if len(df_1d_prev) < 2:
        return np.zeros(n)
    
    high_1d = df_1d_prev['high'].values
    low_1d = df_1d_prev['low'].values
    close_1d = df_1d_prev['close'].values
    
    # Shift to get previous day values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot points
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d_prev, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d_prev, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d_prev, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d_prev, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d_prev, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches S1 or S2 with volume confirmation, 1d uptrend (mean reversion in uptrend)
            if ((low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i]) and
                trend_1d_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 or R2 with volume confirmation, 1d downtrend (mean reversion in downtrend)
            elif ((high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i]) and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches pivot or 1d trend turns down
            if (high[i] >= pivot_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches pivot or 1d trend turns up
            if (low[i] <= pivot_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals