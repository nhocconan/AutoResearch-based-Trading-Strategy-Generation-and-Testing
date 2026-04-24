#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + weekly Camarilla pivot direction + volume confirmation.
- Primary timeframe: 6h targeting 80-150 total trades over 4 years (20-37/year).
- HTF: 1w Camarilla pivot levels (H3/L3) for structure and bias.
- Entry: Long when price breaks above Donchian(20) high AND close > weekly H3 AND volume confirmed.
         Short when price breaks below Donchian(20) low AND close < weekly L3 AND volume confirmed.
- Exit: Opposite Donchian breakout (short breakout for long exit, long breakout for short exit).
- Volume confirmation: current volume > 1.5 * 20-period volume MA.
- Signal size: 0.25 discrete to balance return and drawdown control.
Designed to capture structured moves in both bull and bear markets using institutional pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (H3, L3)
    # Using previous week's OHLC to avoid look-ahead
    wk_high = df_1w['high'].shift(1).values  # Previous week high
    wk_low = df_1w['low'].shift(1).values    # Previous week low
    wk_close = df_1w['close'].shift(1).values # Previous week close
    
    # Camarilla calculations
    pivot = (wk_high + wk_low + wk_close) / 3
    range_wk = wk_high - wk_low
    h3 = pivot + (range_wk * 1.1 / 4)  # H3 level
    l3 = pivot - (range_wk * 1.1 / 4)  # L3 level
    
    # Align weekly H3/L3 to 6h timeframe (completed week only)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Calculate 6h Donchian(20) channels
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Need enough bars for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Donchian breakout above weekly H3 with volume
            if (curr_high > donchian_high[i] and 
                curr_close > h3_aligned[i] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below weekly L3 with volume
            elif (curr_low < donchian_low[i] and 
                  curr_close < l3_aligned[i] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price breaks below Donchian low (short breakout)
            if curr_low < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price breaks above Donchian high (long breakout)
            if curr_high > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wCamarilla_H3L3_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0