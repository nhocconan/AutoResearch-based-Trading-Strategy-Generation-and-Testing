#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Trend_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend (price above/below weekly pivot point) with volume confirmation.
Weekly pivot provides structural support/resistance that works in both bull and bear markets. Donchian breakout captures momentum.
Volume confirmation reduces false breakouts. Designed for 12-30 trades/year on BTC/ETH with controlled risk.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for weekly pivot (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_pivot_vals = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    
    # 1d data for Donchian(20) (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # Donchian(20): 20-period high/low
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d Donchian(20) (20) and session filter
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price above weekly pivot AND breaks above Donchian high with volume confirmation
            long_entry = (curr_close > weekly_pivot_aligned[i]) and (curr_close > donch_high_aligned[i]) and volume_confirm[i]
            # Short: price below weekly pivot AND breaks below Donchian low with volume confirmation
            short_entry = (curr_close < weekly_pivot_aligned[i]) and (curr_close < donch_low_aligned[i]) and volume_confirm[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price breaks below Donchian low or weekly pivot
            if curr_close < donch_low_aligned[i] or curr_close < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian high or weekly pivot
            if curr_close > donch_high_aligned[i] or curr_close > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0