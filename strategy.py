#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses 6h timeframe for signal generation with Donchian channel breakout (20-period)
# Weekly pivot provides higher timeframe directional bias from 1w data
# Volume confirmation (1.8x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Weekly pivot filters for structural bias, reducing counter-trend trades
# Donchian breakout captures momentum, volume confirmation validates breakout strength

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # Pivot = (H+L+C)/3, R1 = 2*P-L, S1 = 2*P-H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Weekly direction: bullish if close > pivot, bearish if close < pivot
    weekly_bullish = close_1w > pivot_1w
    weekly_bearish = close_1w < pivot_1w
    
    # Align weekly direction to 6h timeframe (use previous week's direction)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Donchian channel (20-period) on 6h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price > Donchian upper + weekly bullish + volume confirm
            if close[i] > high_ma[i] and weekly_bullish_aligned[i] > 0.5 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < Donchian lower + weekly bearish + volume confirm
            elif close[i] < low_ma[i] and weekly_bearish_aligned[i] > 0.5 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price < Donchian lower or weekly turns bearish
            if close[i] < low_ma[i] or weekly_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price > Donchian upper or weekly turns bullish
            if close[i] > high_ma[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals