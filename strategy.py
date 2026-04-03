#!/usr/bin/env python3
"""
Experiment #255: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) 
and confirmed by volume spikes capture institutional participation in trend continuations. 
Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. 
Volume confirmation ensures breakouts have conviction. Targets 12-37 trades/year on 6h timeframe 
(50-150 total over 4 years) to minimize fee drag while capturing high-probability trend continuations 
in both bull and bear markets. Uses discrete position sizing (0.25) to balance return and drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Weekly pivot: P = (H + L + C) / 3
        weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume average (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period average volume on 1d
    if len(df_1d) >= 20:
        volume_1d = df_1d['volume'].values
        avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        
        # Align to 6h timeframe
        avg_volume_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20_1d)
    else:
        avg_volume_20_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Donchian Channel(20): upper = max(high,20), lower = min(low,20)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 20)  # Ensure enough data for Donchian and HTF
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(avg_volume_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Current volume > 1.5x 20-day average volume ---
        volume_spike = volume[i] > 1.5 * avg_volume_20_1d_aligned[i]
        
        # --- Weekly Pivot Direction ---
        price_above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > highest_high[i-1]  # Current high exceeds previous period's highest high
        breakout_down = low[i] < lowest_low[i-1]   # Current low exceeds previous period's lowest low
        
        # --- Exit Logic: Stoploss at 2x ATR(14) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + price above weekly pivot + volume spike
        if breakout_up and price_above_weekly_pivot and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Donchian breakout down + price below weekly pivot + volume spike
        elif breakout_down and price_below_weekly_pivot and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals