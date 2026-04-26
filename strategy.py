#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: On 1d timeframe, enter long when price breaks above 20-day Donchian high AND weekly trend is up (close > weekly EMA50) AND volume > 2.0x 20-day average volume. Enter short when price breaks below 20-day Donchian low AND weekly trend is down (close < weekly EMA50) AND volume > 2.0x 20-day average volume. Exit on trend reversal or retracement to Donchian midpoint. Target: 15-30 trades/year. Uses proven Donchian breakout pattern with HTF trend filter and volume confirmation for robustness in both bull and bear markets.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day Donchian channels
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll_max + low_roll_min) / 2
    
    # Volume confirmation: volume > 2.0x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian and EMA warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_roll_max[i]
        breakout_down = close[i] < low_roll_min[i]
        
        # Weekly trend filter
        trend_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: breakout above Donchian high + volume spike + weekly uptrend
            long_signal = breakout_up and volume_spike[i] and trend_uptrend
            
            # Short: breakout below Donchian low + volume spike + weekly downtrend
            short_signal = breakout_down and volume_spike[i] and trend_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend change to downtrend OR price retracing to Donchian midpoint
            if not trend_uptrend or close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend change to uptrend OR price retracing to Donchian midpoint
            if not trend_downtrend or close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0