#!/usr/bin/env python3
"""
1d Weekly Pivot Breakout with Volume Spike and Trend Filter
Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts
above weekly R1 (resistance 1) with volume confirmation and trend filter
capture sustained moves in both bull and bear markets. Low frequency by design.
"""
name = "1d_WeeklyPivot_Breakout_VolumeTrend"
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
    
    # === WEEKLY PIVOT LEVELS (from weekly OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low  # Resistance 1
    weekly_s1 = 2 * weekly_pivot - weekly_high  # Support 1
    
    # Align weekly pivot levels to daily timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === DAILY TREND FILTER (EMA 50) ===
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === VOLUME SPIKE (20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R1 with volume spike and above EMA50
            if (close[i] > r1_aligned[i] and 
                vol_spike[i] and 
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1 with volume spike and below EMA50
            elif (close[i] < s1_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below weekly pivot OR volume dries up
            if close[i] < pivot_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly pivot OR volume dries up
            if close[i] > pivot_aligned[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals