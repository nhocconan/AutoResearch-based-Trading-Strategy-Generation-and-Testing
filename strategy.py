#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Donchian channels identify breakouts from price consolidation
# Weekly pivot (based on prior week's range) provides institutional bias: long above weekly pivot, short below
# Volume confirmation (>1.5x average) ensures breakout legitimacy with strict filtering
# Works in bull/bear: breakouts occur in all regimes, weekly pivot filters counter-trend trades, volume reduces false signals
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_Donchian20_WeeklyPivot_Direction_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Need previous bar's levels to avoid look-ahead
    high_20_prev = np.roll(high_20, 1)
    low_20_prev = np.roll(low_20, 1)
    high_20_prev[0] = np.nan
    low_20_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > high_20_prev
    breakout_down = close < low_20_prev
    
    # Volume confirmation: volume > 1.5x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate weekly pivot points from prior week (using 1w HTF data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot: (Prior week High + Low + Close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (completed weekly bar only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(high_20_prev[i]) or 
            np.isnan(low_20_prev[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and weekly pivot filter
            if curr_volume_confirm:
                # Bullish breakout: price above 6h Donchian high + above weekly pivot
                if curr_breakout_up and curr_close > curr_weekly_pivot:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below 6h Donchian low + below weekly pivot
                elif curr_breakout_down and curr_close < curr_weekly_pivot:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below 6h Donchian low (reversal) or above Donchian high (take profit)
            if curr_close < low_20_prev[i] or curr_close > high_20_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above 6h Donchian high (reversal) or below Donchian low (take profit)
            if curr_close > high_20_prev[i] or curr_close < low_20_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals