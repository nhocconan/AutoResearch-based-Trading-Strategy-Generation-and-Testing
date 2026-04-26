#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_v2
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) and volume confirmation (>1.5x 24-bar average) capture strong trending moves with fewer false signals. Weekly pivot provides structural support/resistance from higher timeframe, reducing whipsaws in ranging markets. Targets 12-25 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    pivot = (wk_high + wk_low + wk_close) / 3.0
    wk_range = wk_high - wk_low
    r1 = 2 * pivot - wk_low
    s1 = 2 * pivot - wk_high
    r2 = pivot + wk_range
    s2 = pivot - wk_range
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Donchian(20) channels on 6h
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.5 * 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian (20), volume MA (24)
    start_idx = max(donchian_window, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(dc_high[i]) or np.isnan(dc_low[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Donchian breakout conditions
        long_breakout = close_val > dc_high[i]
        short_breakout = close_val < dc_low[i]
        
        # Weekly pivot direction filter
        # Long bias: price above weekly pivot and R1
        # Short bias: price below weekly pivot and S1
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        long_filter = close_val > pivot_val and close_val > r1_val
        short_filter = close_val < pivot_val and close_val < s1_val
        
        # Entry conditions: Donchian breakout in direction of weekly pivot bias + volume confirmation
        long_entry = long_breakout and long_filter and volume_confirm[i]
        short_entry = short_breakout and short_filter and volume_confirm[i]
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < pivot_val or close_val < s1_val):
            # Exit long if price breaks below weekly pivot or S1
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > pivot_val or close_val > r1_val):
            # Exit short if price breaks above weekly pivot or R1
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_v2"
timeframe = "6h"
leverage = 1.0