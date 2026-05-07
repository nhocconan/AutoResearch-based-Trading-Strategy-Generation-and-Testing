#!/usr/bin/env python3
# 6H_Donchian20_WeeklyPivot_Direction
# Hypothesis: 6-hour Donchian(20) breakout with weekly pivot direction filter.
# Uses weekly pivot (from 1w data) to determine trend direction: only long when price > weekly pivot, short when price < weekly pivot.
# This avoids counter-trend trades in both bull and bear markets by using higher timeframe structure.
# Targets 15-25 trades/year to minimize fee drag. Uses discrete position sizing (0.25).

name = "6H_Donchian20_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility periods (ATR > 0.2% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.002 * close  # ATR > 0.2% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (1.5x average volume)
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + above weekly pivot + volume filter
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_pivot_aligned[i] and   # Weekly uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below weekly pivot + volume filter
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and   # Weekly downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the opposite Donchian band
            if position == 1 and close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals