#!/usr/bin/env python3

# Hypothesis: 1h timeframe strategy using 4h Donchian breakout for direction and 1h for entry timing.
# Uses volume confirmation and session filter (08-20 UTC) to reduce false signals.
# Target: 15-37 trades/year per symbol by combining higher timeframe structure with lower timeframe precision.
# Works in both bull and bear markets by following 4h trend and requiring volume confirmation.

name = "1h_Donchian20_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

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
    
    # 4h Donchian channels (20-period high/low) for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (wait for 4h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Determine 4h trend based on Donchian breakout
    trend_up = close > donchian_high_aligned
    trend_down = close < donchian_low_aligned
    
    # Volume filter: current volume > 2.0x 24-period average volume
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    # Session filter: 08-20 UTC (use pre-computed hour from index)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend (price above Donchian high) + volume + session
            if trend_up[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (price below Donchian low) + volume + session
            elif trend_down[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend reversal or volume drop
            if not trend_up[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend reversal or volume drop
            if not trend_down[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals