#!/usr/bin/env python3
# 1d Weekly Pivot + Volume + Trend Filter
# Hypothesis: Weekly pivot levels from the 1w chart act as strong support/resistance on the daily chart.
# In trending markets, price pulls back to these levels for continuation entries.
# In ranging markets, reversals occur at these levels.
# Uses 1d timeframe for lower trade frequency (~10-30/year) and combines:
#   - Price touching weekly S1/S2 with volume spike (long)
#   - Price touching weekly R1/R2 with volume spike (short)
#   - EMA50 trend filter to avoid counter-trend trades
# Volume confirmation reduces false signals. Trend filter improves win rate in trends.

name = "1d_WeeklyPivot_Volume_Trend"
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
    
    # === WEEKLY DATA FOR PIVOT LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard calculation)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly levels to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # Strong volume filter to reduce trades
    
    # === TREND FILTER: EMA50 on daily chart ===
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # For EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches S1 or S2 from above with volume, and above EMA50 (uptrend)
            if ((low[i] <= s1_1w_aligned[i] * 1.002 and close[i] > s1_1w_aligned[i]) or
                (low[i] <= s2_1w_aligned[i] * 1.002 and close[i] > s2_1w_aligned[i])) and \
               volume_spike[i] and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 or R2 from below with volume, and below EMA50 (downtrend)
            elif ((high[i] >= r1_1w_aligned[i] * 0.998 and close[i] < r1_1w_aligned[i]) or
                  (high[i] >= r2_1w_aligned[i] * 0.998 and close[i] < r2_1w_aligned[i])) and \
                 volume_spike[i] and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S2 or reaches R2
            if close[i] < s2_1w_aligned[i] or close[i] > r2_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R2 or reaches S2
            if close[i] > r2_1w_aligned[i] or close[i] < s2_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals