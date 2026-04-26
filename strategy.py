#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_12hEMA50_Trend
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high AND volume > 2.0x 20-period average volume AND 12h EMA50 is rising. Enter short when price breaks below 20-period Donchian low AND volume > 2.0x 20-period average volume AND 12h EMA50 is falling. Exit on trend reversal or price retracing to Donchian midpoint. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Target: 19-50 trades/year.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h EMA50 slope (rising/falling)
    ema_slope = np.diff(ema_50_12h, prepend=ema_50_12h[0])
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: fixed threshold of 2.0x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian, EMA, and volume MA warmup
    start_idx = max(lookback, 50, 20)  # Donchian(20), EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_slope_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # 12h trend filter (using EMA slope)
        trend_uptrend = ema_slope_aligned[i] > 0
        trend_downtrend = ema_slope_aligned[i] < 0
        
        if position == 0:
            # Long: breakout above Donchian high + volume spike + 12h uptrend
            long_signal = breakout_up and volume_spike[i] and trend_uptrend
            
            # Short: breakout below Donchian low + volume spike + 12h downtrend
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

name = "4h_Donchian20_Breakout_VolumeSpike_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0