#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_v1
Hypothesis: 6h Donchian(20) breakout traded in direction of weekly Camarilla pivot bias.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when price breaks above 6h Donchian(20) high AND weekly bias is bullish (close > weekly R3)
- Short when price breaks below 6h Donchian(20) low AND weekly bias is bearish (close < weekly S3)
- Weekly Camarilla levels derived from previous 1w OHLC for structural bias
- Donchian breakout provides clean entry/exit with proven edge across market regimes
- Designed for low trade frequency with symmetry between long/short for bear market performance
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1w data ONCE before loop for weekly Camarilla levels (bias filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels from previous 1w bar
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = df_1w['close'].values
    prev_high = df_1w['high'].values
    prev_low = df_1w['low'].values
    
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed 1w bar)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Calculate 6h Donchian channels (20-period)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian channels)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with weekly pivot bias filter
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Weekly bias filters
        weekly_bullish = close[i] > R3_aligned[i]  # Price above weekly R3 = bullish bias
        weekly_bearish = close[i] < S3_aligned[i]  # Price below weekly S3 = bearish bias
        
        if position == 0:
            # Long: bullish breakout AND weekly bullish bias
            if breakout_up and weekly_bullish:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND weekly bearish bias
            elif breakout_down and weekly_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low (reverse signal) OR weekly bias turns bearish
            if breakout_down or not weekly_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high (reverse signal) OR weekly bias turns bullish
            if breakout_up or not weekly_bearish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_v1"
timeframe = "6h"
leverage = 1.0