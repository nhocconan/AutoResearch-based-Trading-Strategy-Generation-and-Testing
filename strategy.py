# 1d_WeeklyPivot_DailyTrend_Volume_v1
# Hypothesis: Use weekly pivots for long-term trend bias, daily price action for entry timing, and volume confirmation to filter breakouts.
# Weekly pivot levels (based on prior week's OHLC) act as strong support/resistance. In an uptrend (daily close > weekly pivot), look for longs on dips to support with volume.
# In a downtrend (daily close < weekly pivot), look for shorts on rallies to resistance with volume.
# This combines trend following (weekly pivot as trend filter) with mean-reversion entries (to daily support/resistance) and volume confirmation.
# Designed for low trade frequency (~10-25/year) to minimize fee drag and work in both bull and bear markets via trend filter.
# Timeframe: 1d, HTF: 1w for weekly pivot calculation.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_DailyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    open_1w = df_1w['open'].values  # though not used in classic pivot, keep for completeness
    
    # Weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(C - L), S3 = L - 2*(H - C)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (close_1w - low_1w)
    s3 = low_1w - 2 * (high_1w - close_1w)
    
    # Align weekly levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Weekly data updates once per week, but we need at least one week
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter: close > pivot = uptrend, close < pivot = downtrend
        is_uptrend = close[i] > pivot_aligned[i]
        
        if position == 0:
            if is_uptrend:
                # In uptrend: look for longs on pullback to daily support (S1) with volume spike
                if low[i] <= s1_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            else:
                # In downtrend: look for shorts on rally to daily resistance (R1) with volume spike
                if high[i] >= r1_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reaches daily resistance (R1) or trend reverses (close < pivot)
            if high[i] >= r1_aligned[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches daily support (S1) or trend reverses (close > pivot)
            if low[i] <= s1_aligned[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals