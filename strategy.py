#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction
Hypothesis: For 6BTC/ETH, combine 6h Donchian(20) breakouts with weekly pivot direction (from 1w data) to filter breakouts. Only take long breakouts when weekly pivot is bullish (price > weekly pivot) and short breakouts when bearish (price < weekly pivot). Uses volume confirmation (volume > 1.5x 20-period average) to avoid false signals. Designed for 15-30 trades/year per symbol to avoid fee drag while capturing major trend moves. Works in bull markets via breakouts and in bear markets via short breakdowns aligned with weekly structure.
"""

name = "6h_Donchian_Breakout_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot and direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- Weekly Pivot (using prior week OHLC) ---
    # Calculate from previous week's OHLC
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    # First bar: use current week's values (no look-ahead)
    prev_week_high[0] = df_1w['high'].values[0]
    prev_week_low[0] = df_1w['low'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # --- 6h Donchian Channels (20-period) ---
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # --- 6h Volume Average for confirmation ---
    vol_avg_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 20  # for Donchian and volume averages
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_avg_6h[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR estimate from 6h range
                atr_est = np.abs(high_6h[i] - low_6h[i])
                if position == 1 and close_6h[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_6h[i] > 1.5 * vol_avg_6h[i]
        
        if position == 0:
            # Look for breakout entries in direction of weekly pivot
            if vol_confirm:
                # Long breakout: price breaks above Donchian high AND above weekly pivot (bullish bias)
                if close_6h[i] > donchian_high[i] and close_6h[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_6h[i]
                # Short breakdown: price breaks below Donchian low AND below weekly pivot (bearish bias)
                elif close_6h[i] < donchian_low[i] and close_6h[i] < weekly_pivot_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_6h[i]
        else:
            # Manage existing position: trail with opposite Donchian band
            if position == 1:
                # Long: exit if price breaks below Donchian low
                if close_6h[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit if price breaks above Donchian high
                if close_6h[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals