#!/usr/bin/env python3
"""
6h Weekly Pivot Donchian Breakout + Volume Spike
Hypothesis: Weekly Camarilla pivot levels (R3/S3, R4/S4) provide significant support/resistance.
Donchian(20) breakout in the direction of weekly pivot (above R3 for long, below S3 for short)
captures institutional breakouts with volume confirmation. Weekly timeframe ensures structural
levels, 6h provides timely execution. Works in bull markets (breakouts above weekly resistance)
and bear markets (breakdowns below weekly support). Targets 12-37 trades/year.
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    # Weekly OHLC from previous completed weekly bar
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Range = H - L
    range_w = weekly_high - weekly_low
    
    # Camarilla levels
    r3 = pp + (range_w * 1.1 / 4.0)  # R3
    s3 = pp - (range_w * 1.1 / 4.0)  # S3
    r4 = pp + (range_w * 1.1 / 2.0)  # R4
    s4 = pp - (range_w * 1.1 / 2.0)  # S4
    
    # Align weekly levels to 6h timeframe (completed weekly bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Donchian(20) channels on 6h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(lookback, 20)  # Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Donchian breakout above R3 AND volume spike
            long_entry = (curr_high > donchian_high[i-1]) and (donchian_high[i-1] > r3_aligned[i]) and vol_spike
            # Short: Donchian breakdown below S3 AND volume spike
            short_entry = (curr_low < donchian_low[i-1]) and (donchian_low[i-1] < s3_aligned[i]) and vol_spike
            
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
            # Exit: Donchian breakdown below midpoint OR loss of weekly structure
            midpoint = (donchian_high[i-1] + donchian_low[i-1]) / 2.0
            if curr_close < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Donchian breakout above midpoint OR loss of weekly structure
            midpoint = (donchian_high[i-1] + donchian_low[i-1]) / 2.0
            if curr_close > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0