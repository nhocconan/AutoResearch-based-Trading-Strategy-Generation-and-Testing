#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_Filter
Hypothesis: On 6h timeframe, Donchian(20) breakouts filtered by weekly trend (price > weekly SMA50 for longs, < for shorts) and volume spike (volume > 1.5 * 20-period average) capture sustained moves with minimal whipsaw. Weekly trend ensures alignment with higher timeframe momentum, reducing false breakouts in ranging markets. Targets 12-30 trades/year (50-120 over 4 years) by requiring confluence of breakout, trend, and volume. Works in both bull (trend-following breakouts) and bear (short breakdowns) regimes.
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
    
    # Donchian(20) channels from prior 20 periods (lookback 20, exclude current)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Weekly trend filter: SMA50 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 20 for Donchian, 50 for weekly SMA, 20 for volume avg
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: price > highest_high + close > weekly SMA50 + volume spike
            long_entry = (close_val > highest_high[i]) and \
                       (close_val > sma_50_1w_aligned[i]) and \
                       volume_spike[i]
            # Short: price < lowest_low + close < weekly SMA50 + volume spike
            short_entry = (close_val < lowest_low[i]) and \
                        (close_val < sma_50_1w_aligned[i]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price breaks below weekly SMA50 (trend change) or reverses to midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close_val < sma_50_1w_aligned[i]) or (close_val < midpoint):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above weekly SMA50 (trend change) or reverses to midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close_val > sma_50_1w_aligned[i]) or (close_val > midpoint):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0