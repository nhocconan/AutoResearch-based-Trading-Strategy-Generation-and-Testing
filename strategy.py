# 6h_DailyPivot_WeeklyTrend_VolumeFilter
# Hypothesis: 6h price breaks above/below 1d pivot (S1/S3/R1/R3) only when weekly trend confirms (price > weekly EMA200 for long, < for short) and volume exceeds 1.5x 6h median volume.
# Uses daily pivot points for intraday mean-reversion/fade and weekly EMA for trend filter to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

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
    
    # Daily pivot points (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate pivots for previous day (to avoid look-ahead)
    # Pivot = (H + L + C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Support/Resistance levels
    s1_1d = 2 * pivot_1d - high_1d
    r1_1d = 2 * pivot_1d - low_1d
    s2_1d = pivot_1d - (high_1d - low_1d)
    r2_1d = pivot_1d + (high_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    
    # Align daily pivot levels to 6h timeframe (already delayed by get_htf_data + align)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Weekly trend filter: EMA200 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume filter: current volume > 1.5x median of last 20 periods
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for weekly EMA200
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long conditions: price breaks above S1 or S2 with weekly uptrend and volume spike
        long_break_s1 = close[i] > s1_1d_aligned[i]
        long_break_s2 = close[i] > s2_1d_aligned[i]
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        volume_spike = volume[i] > vol_threshold[i]
        
        if ((long_break_s1 or long_break_s2) and weekly_uptrend and volume_spike):
            signals[i] = 0.25
        
        # Short conditions: price breaks below R1 or R2 with weekly downtrend and volume spike
        short_break_r1 = close[i] < r1_1d_aligned[i]
        short_break_r2 = close[i] < r2_1d_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        if ((short_break_r1 or short_break_r2) and weekly_downtrend and volume_spike):
            signals[i] = -0.25
        
        # Exit conditions: price returns to pivot level or opposite break
        elif i > 0:
            prev_signal = signals[i-1]
            # Exit long if price falls back below pivot
            if prev_signal == 0.25 and close[i] < pivot_1d_aligned[i]:
                signals[i] = 0.0
            # Exit short if price rises back above pivot
            elif prev_signal == -0.25 and close[i] > pivot_1d_aligned[i]:
                signals[i] = 0.0
            # Otherwise hold position
            else:
                signals[i] = prev_signal
    
    return signals

name = "6h_DailyPivot_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0