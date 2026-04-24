#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot bias and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for weekly pivot bias (trend filter).
- Weekly pivot bias: price above weekly pivot = bullish bias (long only), price below = bearish bias (short only).
- Entry: Long when price breaks above Donchian upper (20-period high) with volume spike and bullish weekly bias.
         Short when price breaks below Donchian lower (20-period low) with volume spike and bearish weekly bias.
- Exit: When price reverts to the Donchian midpoint (mean of upper and lower) or opposite signal.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian channels on 6h (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough weekly bars and Donchian period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Donchian breakout signals with volume spike and weekly pivot bias
            if volume_spike[i]:
                # Long: price breaks above Donchian high with bullish weekly bias (price > weekly pivot)
                if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low with bearish weekly bias (price < weekly pivot)
                elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to Donchian midpoint or short signal
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Donchian midpoint or long signal
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotBias_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0