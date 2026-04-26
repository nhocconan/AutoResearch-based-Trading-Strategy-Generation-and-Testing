#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeFilter
Hypothesis: Donchian(20) breakouts on 12h timeframe with weekly trend filter and volume confirmation capture strong trending moves while avoiding chop. Weekly trend ensures we only trade in the direction of the higher timeframe momentum, reducing false breakouts. Volume confirmation adds conviction to breakouts. Designed to work in both bull and bear markets by being directionally flexible via the weekly trend filter.
Timeframe: 12h, HTF: 1w
Target trades: 12-37/year per symbol (50-150 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian Channel (20-period) on 12h data
    period = 20
    high_roll = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_roll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Volume spike detector (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian channel with volume spike and weekly uptrend
            if close[i] > upper_channel[i] and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian channel with volume spike and weekly downtrend
            elif close[i] < lower_channel[i] and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price re-enters channel (below upper channel) OR weekly trend changes to downtrend
            if close[i] < upper_channel[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price re-enters channel (above lower channel) OR weekly trend changes to uptrend
            if close[i] > lower_channel[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0