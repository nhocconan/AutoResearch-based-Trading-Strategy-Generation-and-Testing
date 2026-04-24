#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w Camarilla pivot structure and volume confirmation.
- Donchian breakout: long when price > highest high of last 20 bars, short when price < lowest low
- 1w Camarilla pivot filter: only take longs when price > weekly H3 level, shorts when price < weekly L3 level
- Volume confirmation: volume > 1.5 * 50-period average to avoid false breakouts
- Designed to capture strong momentum moves aligned with weekly structure in both bull and bear markets
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate 1w Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need enough data for pivot calculation
        return np.zeros(n)
    
    # Weekly OHLC for Camarilla calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla pivot levels: H4, H3, L3, L4
    # H3 = Close + 1.1*(High-Low)/2
    # L3 = Close - 1.1*(High-Low)/2
    camarilla_h3 = weekly_close + 1.1 * (weekly_high - weekly_low) / 2
    camarilla_l3 = weekly_close - 1.1 * (weekly_high - weekly_low) / 2
    
    # Align weekly levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Volume confirmation: volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 50)  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > weekly H3 AND volume confirmation
            if close[i] > highest_high[i] and close[i] > camarilla_h3_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < weekly L3 AND volume confirmation
            elif close[i] < lowest_low[i] and close[i] < camarilla_l3_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR no volume confirmation
            if close[i] < lowest_low[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR no volume confirmation
            if close[i] > highest_high[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wCamarilla_H3L3_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0