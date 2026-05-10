#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: Use 12h Donchian(20) breakout direction filtered by weekly pivot trend (bull/bear) and 6h volume confirmation.
Weekly pivot defines major trend; Donchian breakouts capture momentum in trend direction.
Volume filter avoids false breakouts. Designed for 6-12 trades/year per symbol, works in bull/bear via trend filter.
"""

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_wk = df_1w['high'].shift(1).values
    low_wk = df_1w['low'].shift(1).values
    close_wk = df_1w['close'].shift(1).values
    
    pivot_wk = (high_wk + low_wk + close_wk) / 3.0
    # Weekly R1/S1 for trend direction (not entry levels)
    r1_wk = 2 * pivot_wk - low_wk
    s1_wk = 2 * pivot_wk - high_wk
    
    # Align weekly pivot levels to 6h timeframe
    pivot_wk_aligned = align_htf_to_ltf(prices, df_1w, pivot_wk)
    r1_wk_aligned = align_htf_to_ltf(prices, df_1w, r1_wk)
    s1_wk_aligned = align_htf_to_ltf(prices, df_1w, s1_wk)
    
    # Get 12h data for Donchian breakout calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower bands (20-period high/low)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot, 12h Donchian (20), and volume EMA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_wk_aligned[i]) or 
            np.isnan(r1_wk_aligned[i]) or
            np.isnan(s1_wk_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: price vs weekly pivot
        # Bullish if above weekly pivot, bearish if below
        bullish_trend = close[i] > pivot_wk_aligned[i]
        bearish_trend = close[i] < pivot_wk_aligned[i]
        
        if position == 0:
            # Long: bullish weekly trend AND price breaks above 12h Donchian high with volume
            if bullish_trend and high[i] > donchian_high_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish weekly trend AND price breaks below 12h Donchian low with volume
            elif bearish_trend and low[i] < donchian_low_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 12h Donchian low OR weekly trend turns bearish
            if low[i] < donchian_low_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 12h Donchian high OR weekly trend turns bullish
            if high[i] > donchian_high_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals