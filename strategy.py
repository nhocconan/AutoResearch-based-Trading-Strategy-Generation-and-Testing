#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation (>2x 48-bar average).
- Uses weekly Camarilla H3/L3 from prior completed week for trend bias.
- Enters long when price breaks above 6h Donchian(20) high AND weekly bias bullish (close > weekly H3).
- Enters short when price breaks below 6h Donchian(20) low AND weekly bias bearish (close < weekly L3).
- Volume confirmation requires >2x 48-bar average to ensure conviction.
- Discrete position size 0.25 to manage drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Works in bull/bear: weekly pivot structure adapts to regime; Donchian breakouts capture momentum; volume filter avoids low-conviction entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior 1w OHLC (completed weekly bar)
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # Align to 6h timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate weekly Camarilla levels (H3/L3 for trend bias)
    camarilla_h3_1w = close_1w_aligned + 1.1 * (high_1w_aligned - low_1w_aligned) / 4
    camarilla_l3_1w = close_1w_aligned - 1.1 * (high_1w_aligned - low_1w_aligned) / 4
    
    # 6h Donchian(20) breakout levels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: > 2x 48-period average (48 bars = 12 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 48)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_1w[i]) or np.isnan(camarilla_l3_1w[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout above weekly H3 bias AND volume confirmation
            if close[i] > highest_high[i] and close[i] > camarilla_h3_1w[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below weekly L3 bias AND volume confirmation
            elif close[i] < lowest_low[i] and close[i] < camarilla_l3_1w[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price re-enters Donchian channel OR weekly bias turns bearish
            if close[i] < lowest_low[i] or close[i] < camarilla_l3_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price re-enters Donchian channel OR weekly bias turns bullish
            if close[i] > highest_high[i] or close[i] > camarilla_h3_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyCamarilla_H3L3_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0