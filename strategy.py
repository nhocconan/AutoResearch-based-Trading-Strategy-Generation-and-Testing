#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d Trend + Volume Spike
Hypothesis: Elder Ray (Bull/Bear Power) measures bull/bear strength relative to EMA.
On 6h, we look for extreme power readings with 1d trend alignment and volume spikes.
Works in bull markets via strong bull power + uptrend, in bear via strong bear power + downtrend.
Uses EMA13 for power calculation and EMA45 for trend filter.
Target: 80-150 trades over 4 years (~20-38/year) with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_power_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_prev = np.roll(ema20_1d, 1)
    ema20_1d_prev[0] = ema20_1d[0]
    ema20_rising = ema20_1d > ema20_1d_prev
    ema20_falling = ema20_1d < ema20_1d_prev
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema20_rising_aligned = align_htf_to_ltf(prices, df_1d, ema20_rising)
    ema20_falling_aligned = align_htf_to_ltf(prices, df_1d, ema20_falling)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA13 for power calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Smooth the power signals (EMA8 of raw power)
    bull_power_smooth = pd.Series(bull_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 30  # For EMA13 and EMA8 smoothing
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(ema20_rising_aligned[i]) or np.isnan(ema20_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: power divergence or stoploss
        if position == 1:  # long position
            # Exit: bear power turns positive (selling pressure) OR stoploss
            if (bear_power_smooth[i] > 0 or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turns negative (buying pressure) OR stoploss
            if (bull_power_smooth[i] < 0 or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: extreme power + trend + volume spike
            # Extreme bull power: strong buying pressure
            extreme_bull = bull_power_smooth[i] > np.percentile(bull_power_smooth[max(0, i-50):i+1], 80)
            # Extreme bear power: strong selling pressure
            extreme_bear = bear_power_smooth[i] < np.percentile(bear_power_smooth[max(0, i-50):i+1], 20)
            
            # Volume spike: > 1.8x average
            volume_spike = volume[i] > vol_ema[i] * 1.8
            
            bull_entry = extreme_bull and ema20_rising_aligned[i] and volume_spike
            bear_entry = extreme_bear and ema20_falling_aligned[i] and volume_spike
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals