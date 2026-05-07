#!/usr/bin/env python3
name = "4h_1wPivot_1dEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use previous week's data to calculate this week's pivot
    prev_high = high_1w[:-1]
    prev_low = low_1w[:-1]
    prev_close = close_1w[:-1]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    hl_range = prev_high - prev_low
    r2 = pivot + hl_range
    s2 = pivot - hl_range
    
    # Create arrays for each week (skip first week)
    pivot_per_week = np.full(len(df_1w), np.nan)
    r1_per_week = np.full(len(df_1w), np.nan)
    s1_per_week = np.full(len(df_1w), np.nan)
    r2_per_week = np.full(len(df_1w), np.nan)
    s2_per_week = np.full(len(df_1w), np.nan)
    
    pivot_per_week[1:] = pivot
    r1_per_week[1:] = r1
    s1_per_week[1:] = s1
    r2_per_week[1:] = r2
    s2_per_week[1:] = s2
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_per_week)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_per_week)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_per_week)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_per_week)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_per_week)
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike detection (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_34_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R1, above daily EMA34, volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price < S1, below daily EMA34, volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < S1 or below EMA34
            if (close[i] < s1_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > R1 or above EMA34
            if (close[i] > r1_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h weekly pivot breakout with daily trend filter and volume confirmation.
# Weekly pivot levels (R1/S1) derived from previous week's price action identify key support/resistance.
# Breakout above R1 with volume suggests bullish momentum; breakdown below S1 suggests bearish.
# Daily EMA(34) filter ensures we only trade in the direction of the daily trend.
# Volume confirmation ensures institutional participation.
# Works in bull markets (buy breakouts above R1 in uptrend) and bear markets (sell breakdowns below S1 in downtrend).
# Position size 0.25 balances risk and keeps trade frequency manageable (~15-30 trades/year).