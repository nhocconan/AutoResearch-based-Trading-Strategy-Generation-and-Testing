# 6h_WeeklyPivot_Breakout_Trend
# Hypothesis: Weekly pivot points from weekly data provide strong support/resistance levels. 
# Breakout above weekly R3 or below weekly S3 with volume confirmation and aligned with daily trend.
# Works in bull markets (buy breakouts above R3) and bear markets (sell breakdowns below S3).
# Uses 6h for execution, 1d for trend filter, 1w for pivot levels.
# Target: 60-120 total trades over 4 years (15-30/year) with disciplined entries.

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
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on daily for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly pivot points (standard floor pivot)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Volume average (20-period on 6h)
    vol_avg_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(vol_avg_6h[i])):
            continue
        
        # Long entry: price breaks above weekly R3 + volume spike + price above daily EMA50
        if (close[i] > r3_1w_aligned[i] and
            volume[i] > 1.5 * vol_avg_6h[i] and
            close[i] > ema50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly S3 + volume spike + price below daily EMA50
        elif (close[i] < s3_1w_aligned[i] and
              volume[i] > 1.5 * vol_avg_6h[i] and
              close[i] < ema50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to weekly pivot
        elif position == 1 and close[i] < pivot_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pivot_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Breakout_Trend"
timeframe = "6h"
leverage = 1.0