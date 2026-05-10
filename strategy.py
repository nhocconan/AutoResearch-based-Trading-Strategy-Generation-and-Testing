#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_VolumeConfirm_v2
# Hypothesis: Combines weekly pivot points with 6h Donchian breakouts and volume confirmation.
# Long when price breaks above 20-period Donchian high with volume > 1.5x average and price above weekly pivot.
# Short when price breaks below 20-period Donchian low with volume > 1.5x average and price below weekly pivot.
# Uses weekly pivot for trend filter to avoid counter-trend trades. Designed for 15-30 trades/year.

name = "6h_WeeklyPivot_DonchianBreakout_VolumeConfirm_v2"
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
    
    # Calculate weekly pivot points (using weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: P = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian high with volume confirmation and above weekly pivot
            if close[i] > donchian_high[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian low with volume confirmation and below weekly pivot
            elif close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < weekly_pivot_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below weekly pivot
            if close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above weekly pivot
            if close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals