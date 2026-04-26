#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeSpike_v1
Hypothesis: 6h Donchian(20) breakout with weekly Camarilla pivot trend filter and volume spike confirmation.
Enters long when price breaks above 20-period high AND weekly trend is bullish (close > weekly H3) AND volume spike.
Enters short when price breaks below 20-period low AND weekly trend is bearish (close < weekly L3) AND volume spike.
Uses weekly Camarilla H3/L3 for higher timeframe trend alignment to avoid counter-trend trades.
Volume spike confirms institutional participation. Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in bull/bear markets by trading with the weekly trend and using volume to filter false breakouts.
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla levels: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    camarilla_h3_1w = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_l3_1w = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align weekly Camarilla levels to 6h timeframe (use previous week's levels)
    camarilla_h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(camarilla_h3_1w_aligned[i]) or np.isnan(camarilla_l3_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > high_20[i]
        breakout_short = close[i] < low_20[i]
        
        # Weekly trend filter: bullish if close > weekly H3, bearish if close < weekly L3
        weekly_bullish = close[i] > camarilla_h3_1w_aligned[i]
        weekly_bearish = close[i] < camarilla_l3_1w_aligned[i]
        
        if position == 0:
            # Long: bullish breakout AND weekly bullish trend AND volume spike
            if breakout_long and weekly_bullish and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND weekly bearish trend AND volume spike
            elif breakout_short and weekly_bearish and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: bearish breakout OR weekly trend turns bearish
            if breakout_short or not weekly_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish breakout OR weekly trend turns bullish
            if breakout_long or weekly_bullish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0