#!/usr/bin/env python3
# 6H_WEEKLY_PIVOT_BREAKOUT_VOLUME
# Hypothesis: Weekly pivot points provide key support/resistance levels for 6h timeframe.
# Breakouts above weekly R1/R2 with volume confirmation indicate bullish momentum,
# breakdowns below weekly S1/S2 with volume confirmation indicate bearish momentum.
# Weekly trend filter (price vs weekly EMA20) ensures alignment with higher timeframe trend.
# Designed to work in both bull and bear markets by following the weekly trend direction.
# Targets 25-40 trades/year to minimize fee drain with high-probability breakout setups.

name = "6H_WEEKLY_PIVOT_BREAKOUT_VOLUME"
timeframe = "6h"
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
    
    # Weekly high, low, close for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Weekly EMA20 for trend filter
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Volume confirmation: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(weekly_ema20_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and above weekly EMA20 (uptrend)
            if close[i] > r1_aligned[i] and vol_confirm and close[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation and below weekly EMA20 (downtrend)
            elif close[i] < s1_aligned[i] and vol_confirm and close[i] < weekly_ema20_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below pivot or trend breaks
            if close[i] < pivot_aligned[i] or close[i] < weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above pivot or trend breaks
            if close[i] > pivot_aligned[i] or close[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals