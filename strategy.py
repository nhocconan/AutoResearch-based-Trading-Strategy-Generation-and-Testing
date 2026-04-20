#!/usr/bin/env python3
# 6h_WeeklyPivot_R1_S1_Breakout_TrendFilter
# Hypothesis: Weekly R1/S1 breakouts on 6h timeframe with trend filter capture institutional moves while avoiding counter-trend entries.
# Uses weekly pivot points from actual 1w data. Trend filter uses 6h EMA50 to ensure trades align with medium-term direction.
# Works in bull markets by catching breaks above R1 in uptrend; in bear markets by catching breaks below S1 in downtrend.
# Target: 15-30 trades/year to minimize fee drag.

name = "6h_WeeklyPivot_R1_S1_Breakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = pivot + (high_1w - low_1w) * 1.1 / 12
    s1 = pivot - (high_1w - low_1w) * 1.1 / 12
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Trend filter: EMA50 on 6h timeframe
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + in uptrend (price > EMA50) + volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + in downtrend (price < EMA50) + volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (mean reversion to weekly support)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (mean reversion to weekly resistance)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals