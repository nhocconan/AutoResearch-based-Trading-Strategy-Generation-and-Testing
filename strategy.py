#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout in direction of weekly Camarilla pivot bias (R4/S4) with volume confirmation (>1.5x 20-period MA). 
Weekly pivot defines regime: price above weekly R4 = bullish bias (long breakouts only), below weekly S4 = bearish bias (short breakouts only).
Volume confirmation filters low-momentum breakouts. Designed to work in both bull and bear markets by following weekly pivot bias.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

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
    
    # Get weekly data for Camarilla pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from prior week OHLC
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift by 1 to use prior week's OHLC
    close_1w_prev = np.roll(close_1w, 1)
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev[0] = np.nan
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    
    # Weekly Camarilla R4, S4 levels (breakout levels)
    camarilla_range_1w = high_1w_prev - low_1w_prev
    r4 = close_1w_prev + camarilla_range_1w * 1.1 / 2
    s4 = close_1w_prev - camarilla_range_1w * 1.1 / 2
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Weekly pivot bias: above R4 = bullish, below S4 = bearish
    weekly_bullish = close > r4_aligned
    weekly_bearish = close < s4_aligned
    
    # 6h Donchian(20) breakout levels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian + 20 for volume MA)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
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
            # Long: price breaks above Donchian high with weekly bullish bias and volume spike
            if (close[i] > donchian_high[i] and 
                weekly_bullish[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with weekly bearish bias and volume spike
            elif (close[i] < donchian_low[i] and 
                  weekly_bearish[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Donchian low (breakdown) OR weekly bias turns bearish
            if (close[i] < donchian_low[i] or not weekly_bullish[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Donchian high (breakout) OR weekly bias turns bullish
            if (close[i] > donchian_high[i] or not weekly_bearish[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0