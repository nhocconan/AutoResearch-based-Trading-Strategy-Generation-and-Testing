#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot
direction (price above/below weekly pivot) and volume confirmation capture
strong momentum moves. Weekly pivot provides structural bias from higher timeframe
(1w), reducing false breakouts in sideways markets. Works in both bull/bear
markets by only taking breakouts in direction of weekly bias. Targets 12-30
trades/year to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe (use previous week's pivot)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot, additional_delay_bars=1)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR(14) for volatility filtering and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup periods
    start_idx = max(lookback, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        pivot_level = weekly_pivot_aligned[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot (bullish bias)
            long_breakout = curr_high > donchian_high
            long_bias = curr_close > pivot_level
            long_condition = long_breakout and long_bias and vol_spike
            
            # Short: price breaks below Donchian low AND below weekly pivot (bearish bias)
            short_breakout = curr_low < donchian_low
            short_bias = curr_close < pivot_level
            short_condition = short_breakout and short_bias and vol_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Stoploss: 2.5 * ATR below entry
            if curr_close <= entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit: price returns below Donchian high or breaks below weekly pivot
            elif curr_close < donchian_high or curr_close < pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Stoploss: 2.5 * ATR above entry
            if curr_close >= entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit: price returns above Donchian low or breaks above weekly pivot
            elif curr_close > donchian_low or curr_close > pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0