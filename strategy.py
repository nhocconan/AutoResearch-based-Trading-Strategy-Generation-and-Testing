#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) capture strong trending moves while avoiding counter-trend whipsaw. Weekly pivot provides structural support/resistance from higher timeframe, improving breakout quality in both bull and bear markets. Volume confirmation (>1.5x 20-bar average) ensures breakout legitimacy. Discrete sizing (0.25) targets 12-30 trades/year to minimize fee drag. ATR-based stoploss (2.0 ATR) controls drawdown.
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
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of Donchian (20), ATR (14), volume MA (20)
    start_idx = max(lookback, 14, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        pivot_val = weekly_pivot_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(pivot_val) or np.isnan(atr_val) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Donchian breakout conditions
        long_breakout = close_val > highest_high[i]
        short_breakout = close_val < lowest_low[i]
        
        # Weekly pivot direction filter
        is_bullish_bias = close_val > pivot_val  # Price above weekly pivot = bullish
        is_bearish_bias = close_val < pivot_val  # Price below weekly pivot = bearish
        
        # Entry conditions: Donchian breakout in direction of weekly pivot bias + volume
        long_entry = long_breakout and is_bullish_bias and vol_conf
        short_entry = short_breakout and is_bearish_bias and vol_conf
        
        # Exit conditions: ATR-based stoploss or opposite Donchian touch
        long_exit = False
        short_exit = False
        if position == 1:
            # Long stoploss: entry price - 2.0 * ATR
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < stop_price or close_val < lowest_low[i]  # Stop or Donchian breakdown
        elif position == -1:
            # Short stoploss: entry price + 2.0 * ATR
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > stop_price or close_val > highest_high[i]  # Stop or Donchian breakout
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Approximate entry price for stop calculation
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0