#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with Volume Confirmation
# Long when price breaks above weekly pivot R1 + volume > 1.5x 20-period avg
# Short when price breaks below weekly pivot S1 + volume > 1.5x 20-period avg
# Weekly pivots derived from 1w OHLC provide institutional support/resistance.
# Volume filter ensures breakout legitimacy. Designed for low trade frequency (12-30/year).
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by capturing institutional level breaks.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1w Indicator: Weekly Pivot Points (R1, S1) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot: P = (H + L + C) / 3
    # R1 = (2*P) - L, S1 = (2*P) - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2.0 * pivot_1w) - low_1w
    s1_1w = (2.0 * pivot_1w) - high_1w
    
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above weekly pivot R1
        # 2. Volume confirmation
        if (close[i] > r1_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below weekly pivot S1
        # 2. Volume confirmation
        elif (close[i] < s1_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WeeklyPivotR1S1_Breakout_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0