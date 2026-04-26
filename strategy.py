#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_TrendAlign_v1
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend (price > weekly pivot = long bias, price < weekly pivot = short bias) with volume confirmation. Weekly pivot provides structural support/resistance from higher timeframe. Donchian breakout captures momentum. Volume filter ensures breakout legitimacy. Works in bull/bear via pivot trend filter: in bull markets price stays above weekly pivot favoring longs, in bear markets price stays below favoring shorts. Targets 50-150 trades over 4 years by requiring breakout + pivot alignment + volume spike. Discrete position sizing (0.25) minimizes fee churn.
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
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula: P = (H+L+C)/3)
    # Using previous week's OHLC for current week's pivot (no look-ahead)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume spike filter (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA)
    start_idx = donchian_period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions with volume confirmation
        bullish_breakout = (close[i] > highest_high[i-1]) and volume_spike[i]
        bearish_breakout = (close[i] < lowest_low[i-1]) and volume_spike[i]
        
        # Weekly pivot trend filter
        above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Exit conditions: reverse breakout or loss of pivot alignment
        exit_long = (position == 1) and ((close[i] < lowest_low[i]) or (below_weekly_pivot and not bullish_breakout))
        exit_short = (position == -1) and ((close[i] > highest_high[i]) or (above_weekly_pivot and not bearish_breakout))
        
        if exit_long or exit_short:
            signals[i] = 0.0
            position = 0
        elif bullish_breakout and above_weekly_pivot:
            # Long breakout above weekly pivot (bullish alignment)
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif bearish_breakout and below_weekly_pivot:
            # Short breakout below weekly pivot (bearish alignment)
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_TrendAlign_v1"
timeframe = "6h"
leverage = 1.0