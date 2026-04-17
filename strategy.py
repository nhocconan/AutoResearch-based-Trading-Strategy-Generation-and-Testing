#!/usr/bin/env python3
"""
Hypothesis: 12h Elder Ray Index with Weekly Trend Filter and Volume Spike.
Long when Elder Bull Power > 0, Bear Power < 0, volume > 1.5x average, and weekly EMA50 rising.
Short when Elder Bull Power < 0, Bear Power > 0, volume > 1.5x average, and weekly EMA50 falling.
Exit when Elder Bull Power and Bear Power converge (|Bull - Bear| < 0.1 * price) or volume drops.
Uses 1d for Elder Ray calculation (high/low/close), 12h for price/volume, 1w for weekly trend filter.
Target: 50-150 total trades over 4 years (12-37/year). Uses strict confluence to avoid overtrading.
"""

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
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d Elder Ray Index: Bull Power = High - EMA13, Bear Power = Low - EMA13
    def calculate_ema(values, period):
        ema = np.zeros_like(values)
        if len(values) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema13_1d = calculate_ema(close_1d, 13)
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = calculate_ema(close_1w, 50)
    weekly_rising = ema50_1w > np.roll(ema50_1w, 1)  # current > previous
    weekly_falling = ema50_1w < np.roll(ema50_1w, 1)  # current < previous
    
    # Align 1d indicators to 12h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Align weekly indicators to 12h timeframe
    weekly_rising_aligned = align_htf_to_ltf(prices, df_1w, weekly_rising.astype(float))
    weekly_falling_aligned = align_htf_to_ltf(prices, df_1w, weekly_falling.astype(float))
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(weekly_rising_aligned[i]) or 
            np.isnan(weekly_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        bull = bull_power_1d_aligned[i]
        bear = bear_power_1d_aligned[i]
        weekly_up = weekly_rising_aligned[i] > 0.5
        weekly_down = weekly_falling_aligned[i] > 0.5
        
        # Convergence condition: |Bull - Bear| < 0.1 * price (exit signal)
        convergence = abs(bull - bear) < (0.1 * price)
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, volume spike, weekly rising
            if bull > 0 and bear < 0 and vol_spike and weekly_up:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, volume spike, weekly falling
            elif bull < 0 and bear > 0 and vol_spike and weekly_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: convergence OR weekly trend turns down
            if convergence or weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: convergence OR weekly trend turns up
            if convergence or weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ElderRay_VolumeSpike_WeeklyTrend"
timeframe = "12h"
leverage = 1.0