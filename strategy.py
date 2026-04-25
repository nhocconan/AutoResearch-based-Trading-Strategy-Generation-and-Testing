#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDir_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout filtered by weekly pivot direction (from 1w) and volume confirmation (>2.0x 20-period average).
Weekly pivot provides structural bias: price above weekly pivot = long bias, below = short bias.
This reduces false breakouts and aligns with higher timeframe structure.
Targets 12-30 trades/year (50-120 over 4 years) by requiring confluence of breakout, pivot alignment, and volume spike.
Works in both bull/bear: weekly pivot adapts to regime, volume confirms conviction.
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
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 6h data for Donchian(20) (loaded ONCE)
    df_6h = get_htf_data(prices, '6h')
    donch_high = pd.Series(df_6h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_6h['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    
    # 1w data for weekly pivot (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) (20) and 1w pivot (1)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Look for entry signals: Donchian breakout with weekly pivot alignment and volume confirmation
            # Long: price breaks above Donchian(20) high, above weekly pivot (bullish bias), with volume confirmation
            long_breakout = (curr_close > donch_high_aligned[i]) and (curr_close > weekly_pivot_aligned[i]) and volume_confirm[i]
            # Short: price breaks below Donchian(20) low, below weekly pivot (bearish bias), with volume confirmation
            short_breakout = (curr_close < donch_low_aligned[i]) and (curr_close < weekly_pivot_aligned[i]) and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below Donchian(20) low (structure break) or below weekly pivot (bias change)
            if curr_close < donch_low_aligned[i] or curr_close < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above Donchian(20) high (structure break) or above weekly pivot (bias change)
            if curr_close > donch_high_aligned[i] or curr_close > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDir_VolumeConfirm"
timeframe = "6h"
leverage = 1.0