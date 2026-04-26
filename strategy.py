#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation_v1
Hypothesis: 6h Donchian(20) breakout in direction of weekly Camarilla pivot trend with volume confirmation.
Long when price breaks above 20-bar high AND weekly trend is bullish (close > weekly R1) with volume spike.
Short when price breaks below 20-bar low AND weekly trend is bearish (close < weekly S1) with volume spike.
Uses weekly Camarilla R1/S1 as trend filter (more robust than simple EMA) and Donchian for breakout timing.
Designed to work in both bull and bear markets by following weekly pivot trend.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla R1, S1 from prior weekly OHLC
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift by 1 to use prior week's OHLC for current week's levels
    close_1w_prev = np.roll(close_1w, 1)
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev[0] = np.nan
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    
    # Weekly Camarilla R1, S1 levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_range = high_1w_prev - low_1w_prev
    r1_weekly = close_1w_prev + camarilla_range * 1.1 / 12
    s1_weekly = close_1w_prev - camarilla_range * 1.1 / 12
    
    # Align weekly Camarilla levels to 6h timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1w, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1w, s1_weekly)
    
    # 6h Donchian(20) channels
    # Use rolling window with min_periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 30-period MA (tight threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for volume MA + 20 for Donchian + 1 for weekly shift)
    start_idx = 51
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly bullish trend (close > weekly R1) with volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > r1_weekly_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND weekly bearish trend (close < weekly S1) with volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < s1_weekly_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Donchian low (breakdown) OR weekly trend turns bearish (close < weekly S1)
            if (close[i] < donchian_low[i] or close[i] < s1_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Donchian high (breakout) OR weekly trend turns bullish (close > weekly R1)
            if (close[i] > donchian_high[i] or close[i] > r1_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0