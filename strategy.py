#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d pivot direction filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for pivot bias and trend context.
- Daily pivot levels calculated from previous 1d OHLC (standard floor pivot: P=(H+L+C)/3).
- Bias: Long only when close > daily pivot, short only when close < daily pivot.
- Entry: Long when price breaks above Donchian(20) high with volume spike and bullish bias.
         Short when price breaks below Donchian(20) low with volume spike and bearish bias.
- Exit: When price returns to the Donchian(20) midpoint (mean reversion edge) or opposite breakout.
- Works in bull via buying breakouts in uptrend bias, in bear via selling breakdowns in downtrend bias.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_standard_pivot(high, low, close):
    """Calculate standard floor pivot levels for given OHLC"""
    pivot = (high + low + close) / 3.0
    return pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot bias and Donchian context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d standard pivot for bias
    pivot_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        pivot_1d[i] = calculate_standard_pivot(
            df_1d['high'].iloc[i],
            df_1d['low'].iloc[i],
            df_1d['close'].iloc[i]
        )
    
    # Calculate 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Align 1d pivot to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Need enough bars for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and pivot bias
            if volume_spike[i]:
                # Bullish breakout: price > Donchian high and close > daily pivot
                if close[i] > highest_high[i] and close[i] > pivot_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < Donchian low and close < daily pivot
                elif close[i] < lowest_low[i] and close[i] < pivot_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midpoint (mean reversion) or opposite breakdown
            if close[i] <= donchian_mid[i] or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midpoint (mean reversion) or opposite breakout
            if close[i] >= donchian_mid[i] or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dPivotBias_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0