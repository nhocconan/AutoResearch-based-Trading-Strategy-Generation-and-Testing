#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot structure and daily trend filter
# Long when price breaks above weekly R1 with 1d EMA34 > EMA89 and volume > 1.5x average
# Short when price breaks below weekly S1 with 1d EMA34 < EMA89 and volume > 1.5x average
# Weekly pivot levels calculated from prior week's OHLC
# Target: 50-150 total trades over 4 years (12-37/year) with strict entry conditions
# Works in bull (breakout continuation) and bear (mean reversion at extremes)

name = "6h_WeeklyPivot_1dEMA_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Weekly R1: 2*P - L
    r1_w = 2 * pivot_w - low_w
    # Weekly S1: 2*P - H
    s1_w = 2 * pivot_w - high_w
    
    # Get daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA34 and EMA89 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_above_89 = ema34_1d > ema89_1d
    
    # Align weekly pivot levels to 6h
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Align daily EMA trend to 6h
    ema34_above_89_aligned = align_htf_to_ltf(prices, df_1d, ema34_above_89.astype(float))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 90  # Ensure enough data for EMA89
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or
            np.isnan(ema34_above_89_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1, daily EMA34 > EMA89, volume spike
            if (close[i] > r1_w_aligned[i] and
                ema34_above_89_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: price breaks below weekly S1, daily EMA34 < EMA89, volume spike
            elif (close[i] < s1_w_aligned[i] and
                  not ema34_above_89_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: price breaks below weekly S1 or max 10 bars held (~2.5 days)
            if (close[i] < s1_w_aligned[i] or
                i - entry_bar >= 10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly R1 or max 10 bars held
            if (close[i] > r1_w_aligned[i] or
                i - entry_bar >= 10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals