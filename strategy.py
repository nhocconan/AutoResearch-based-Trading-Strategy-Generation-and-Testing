#!/usr/bin/env python3
"""
6h Weekly Pivot + Donchian(20) Breakout with Volume Confirmation
Hypothesis: Weekly pivot levels (from 1w data) act as strong support/resistance. 
When price breaks Donchian(20) channels in the direction of the weekly pivot bias 
(R1 for long, S1 for short) and is confirmed by volume spikes, it captures 
institutional breakout moves. Works in both bull (break above R1) and bear 
(break below S1) regimes. Designed for 6h timeframe to target 12-37 trades/year 
(50-150 over 4 years) by requiring confluence of weekly pivot bias, Donchian 
breakout, and volume confirmation, minimizing fee drag.
"""

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
    
    # Load 1w data ONCE before loop for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    # Using previous week's OHLC to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    weekly_high_prev = np.roll(weekly_high, 1)
    weekly_low_prev = np.roll(weekly_low, 1)
    weekly_close_prev = np.roll(weekly_close, 1)
    # First value will be invalid (rolled from last), set to NaN
    weekly_high_prev[0] = np.nan
    weekly_low_prev[0] = np.nan
    weekly_close_prev[0] = np.nan
    
    weekly_pivot = (weekly_high_prev + weekly_low_prev + weekly_close_prev) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low_prev
    weekly_s1 = 2 * weekly_pivot - weekly_high_prev
    
    # Align weekly pivot levels to 6h timeframe (only use after weekly bar closes)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian(20) channels on primary timeframe (6h)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20) and weekly data
    start_idx = max(20, 1)  # Donchian lookback, weekly data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Weekly pivot bias: price above/below pivot
        bullish_bias = curr_close > weekly_pivot_aligned[i]
        bearish_bias = curr_close < weekly_pivot_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: 
            # Weekly pivot bias (R1 for long, S1 for short) + Donchian breakout + volume spike
            long_entry = (curr_high > donchian_upper[i]) and bullish_bias and (curr_close > weekly_r1_aligned[i]) and vol_spike
            short_entry = (curr_low < donchian_lower[i]) and bearish_bias and (curr_close < weekly_s1_aligned[i]) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below weekly pivot (loss of bullish bias) OR Donchian lower (mean reversion)
            if (curr_close < weekly_pivot_aligned[i]) or (curr_low < donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above weekly pivot (loss of bearish bias) OR Donchian upper (mean reversion)
            if (curr_close > weekly_pivot_aligned[i]) or (curr_high > donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0