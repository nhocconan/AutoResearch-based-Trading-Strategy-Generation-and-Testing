#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses discrete sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years (12-37/year).
# Weekly pivot from 1w data provides structural bias: long only above weekly pivot, short only below.
# Volume confirmation filters breakouts. Works in bull markets (breakouts with trend) and bear markets
# (breakouts against trend filtered out by weekly pivot). Focus on BTC/ETH as primary symbols.

name = "6h_Donchian20_WeeklyPivot_VolumeSpike_v1"
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
    
    # Calculate 6h Donchian(20) from prior 6h bar (breakout of prior 20-period channel)
    prior_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    prior_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate weekly pivot from 1w data (using prior completed weekly bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(prior_20_high[i]) or np.isnan(prior_20_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donch_high = prior_20_high[i]
        curr_donch_low = prior_20_low[i]
        curr_weekly_pivot = weekly_pivot_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and weekly pivot filter
            if curr_volume_spike:
                # Bullish: Close breaks above Donchian high + price above weekly pivot
                if curr_close > curr_donch_high and curr_close > curr_weekly_pivot:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below Donchian low + price below weekly pivot
                elif curr_close < curr_donch_low and curr_close < curr_weekly_pivot:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: close drops below Donchian low OR loses weekly pivot bias
            if curr_close < curr_donch_low or curr_close < curr_weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: close rises above Donchian high OR loses weekly pivot bias
            if curr_close > curr_donch_high or curr_close > curr_weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals