#!/usr/bin/env python3
"""
12h_Donchian20_1dTrend_VolumeBreakout
Hypothesis: 12h Donchian(20) breakout with 1d trend filter (price >/<- EMA50) and volume confirmation (>1.5x 20-bar avg). Enters long when price breaks above upper Donchian channel in 1d uptrend with volume spike, short when breaks below lower channel in 1d downtrend with volume spike. Uses discrete sizing (0.25) to limit fee churn. Designed for 12h timeframe with ~12-30 trades/year, works in bull/bear by following 1d trend filter.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) - using rolling window with min_periods
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 20-period data for Donchian and 50 for 1d EMA
    start_idx = max(period, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian in 1d uptrend with volume confirmation
            long_setup = (close[i] > highest_high[i]) and (close_1d[i] > ema_50_1d_aligned[i]) and volume_spike[i]
            # Short: price breaks below lower Donchian in 1d downtrend with volume confirmation
            short_setup = (close[i] < lowest_low[i]) and (close_1d[i] < ema_50_1d_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian OR trend turns down
            if (close[i] < lowest_low[i]) or (close_1d[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian OR trend turns up
            if (close[i] > highest_high[i]) or (close_1d[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_1dTrend_VolumeBreakout"
timeframe = "12h"
leverage = 1.0