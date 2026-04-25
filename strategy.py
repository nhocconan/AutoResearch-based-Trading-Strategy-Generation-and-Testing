#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend
Hypothesis: 4h Donchian(20) breakout with volume spike (>2.0x 20-bar avg) and 12h EMA50 trend filter. 
Enters long when price breaks above upper Donchian channel with volume spike and 12h uptrend (price > EMA50), 
short when price breaks below lower Donchian channel with volume spike and 12h downtrend (price < EMA50). 
Uses discrete position sizing (0.30) to limit fee churn. Designed for 4h timeframe with ~25-40 trades/year.
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
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20), EMA50, and volume MA
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and 12h uptrend
            long_setup = (close[i] > high_roll[i]) and volume_spike[i] and (close[i] > ema_50_12h_aligned[i])
            # Short: price breaks below lower Donchian with volume spike and 12h downtrend
            short_setup = (close[i] < low_roll[i]) and volume_spike[i] and (close[i] < ema_50_12h_aligned[i])
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price breaks below lower Donchian OR trend turns down
            if (close[i] < low_roll[i]) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price breaks above upper Donchian OR trend turns up
            if (close[i] > high_roll[i]) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend"
timeframe = "4h"
leverage = 1.0