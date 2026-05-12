#!/usr/bin/env python3
# 6h_PivotRange_MeanReversion_1dTrend
# Hypothesis: Mean reversion within weekly pivot range (S2 to R2) with 1d trend filter.
# In ranging markets, price tends to revert to weekly pivot; in trending markets,
# we only take trades in direction of 1d EMA50. Designed for low frequency and robustness.

name = "6h_PivotRange_MeanReversion_1dTrend"
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
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Weekly pivot levels (S2, R2) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    wk_high_prev = np.roll(wk_high, 1)
    wk_low_prev = np.roll(wk_low, 1)
    wk_close_prev = np.roll(wk_close, 1)
    wk_high_prev[0] = np.nan
    wk_low_prev[0] = np.nan
    wk_close_prev[0] = np.nan
    
    pivot = (wk_high_prev + wk_low_prev + wk_close_prev) / 3.0
    r2 = pivot + (wk_high_prev - wk_low_prev)
    s2 = pivot - (wk_high_prev - wk_low_prev)
    
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # === Weekly range width for dynamic sizing ===
    range_width = r2_aligned - s2_aligned
    # Avoid division by zero
    range_width = np.where(range_width == 0, np.nan, range_width)
    
    # Position within weekly range (0 = S2, 1 = R2)
    pos_in_range = (close - s2_aligned) / range_width
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(pos_in_range[i]) or np.isnan(range_width[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Mean reversion signals within weekly range
        # Long near S2 (oversold), Short near R2 (overbought)
        long_signal = pos_in_range[i] < 0.3  # Near support
        short_signal = pos_in_range[i] > 0.7  # Near resistance
        
        if position == 0:
            # In ranging market: take both directions
            # In trending market: only take trades in trend direction
            if long_signal and trend_up:
                signals[i] = 0.25
                position = 1
            elif short_signal and trend_down:
                signals[i] = -0.25
                position = -1
            # Also take counter-trend at extremes if trend is weak (price near extreme but not strong trend)
            elif long_signal and not trend_down:  # Not in strong downtrend
                signals[i] = 0.20
                position = 1
            elif short_signal and not trend_up:  # Not in strong uptrend
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: near R2 or trend reversal to down
            if short_signal or trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: near S2 or trend reversal to up
            if long_signal or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals