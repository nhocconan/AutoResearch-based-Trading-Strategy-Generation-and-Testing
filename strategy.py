#!/usr/bin/env python3
"""
1d_Donchian20_1wTrend_Filter_VolumeConfirmation
Hypothesis: Daily Donchian(20) breakout with weekly trend filter (price above/below weekly EMA50) and volume confirmation (>2.0x 20-bar avg). 
Enters long when price breaks above upper Donchian in weekly uptrend with volume spike, short when breaks below lower Donchian in weekly downtrend with volume spike. 
Exits on opposite Donchian touch or trend reversal. 
Designed for 1d timeframe with ~10-25 trades/year, works in bull/bear by following weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels for 1d timeframe using previous 20 bars
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 20 bars of previous data and EMA50 warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian in weekly uptrend with volume confirmation
            long_setup = (close[i] > donchian_upper[i]) and (close[i] > ema_50_1w_aligned[i]) and volume_spike[i]
            # Short: price breaks below lower Donchian in weekly downtrend with volume confirmation
            short_setup = (close[i] < donchian_lower[i]) and (close[i] < ema_50_1w_aligned[i]) and volume_spike[i]
            
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
            # Exit: price touches lower Donchian OR trend turns down
            if (close[i] <= donchian_lower[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian OR trend turns up
            if (close[i] >= donchian_upper[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_1wTrend_Filter_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0