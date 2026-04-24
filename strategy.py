#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and 1d volume confirmation.
- Primary timeframe: 6h for execution.
- HTF: 1w for pivot points (direction), 1d for volume confirmation.
- Donchian breakout: price breaks above 20-period high (long) or below 20-period low (short).
- Weekly pivot: calculates weekly pivot point (PP) and bias (price > PP = bullish bias, price < PP = bearish bias).
- Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA.
- Entry: Long when price breaks above Donchian(20) high AND weekly bias bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND weekly bias bearish AND volume spike.
- Exit: When price breaks opposite Donchian level (e.g., long exits when price breaks below Donchian low).
- Works in bull via buying breakouts with uptrend bias, in bear via selling breakdowns with downtrend bias.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot point: PP = (High + Low + Close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Bias: 1 = bullish (price > PP), -1 = bearish (price < PP)
    weekly_bias = np.where(weekly_close > weekly_pp, 1, -1)
    
    # Align weekly bias to 6h
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Get 1d data for volume confirmation (using 1d volume for stability)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period MA
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels on 6h: 20-period high/low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period 1d volume MA
        # Note: using 1d volume for less noise; aligns to 6b via align_htf_to_ltf
        # We need current 1d volume - get it from aligned 1d volume series (but we need raw volume aligned)
        # Instead, get 1d volume and align it
        if i == start_idx:  # inefficient but clear; better to precompute
            pass
        
        # Precompute aligned 1d volume outside loop for efficiency
        # We'll do it inside for now but note: should precompute
        
    # Re-structure: precompute all aligned arrays before loop
    
    # Extract and align 1d volume
    vol_1d_raw = df_1d['volume'].values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_raw)
    vol_ma_1d_raw = pd.Series(vol_1d_raw).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d_raw)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period 1d volume MA
        vol_spike = vol_1d_aligned[i] > (1.5 * vol_ma_1d_aligned[i])
        
        if position == 0:
            # Check for Donchian breakout with weekly bias and volume spike
            if vol_spike:
                # Long: price breaks above Donchian high AND weekly bias bullish
                if close[i] > highest_high[i] and weekly_bias_aligned[i] == 1:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low AND weekly bias bearish
                elif close[i] < lowest_low[i] and weekly_bias_aligned[i] == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1wPivotBias_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0