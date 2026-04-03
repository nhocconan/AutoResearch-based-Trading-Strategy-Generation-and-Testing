#!/usr/bin/env python3
"""
Experiment #1879: 6h Donchian Breakout + 12h Pivot Direction + Volume Spike
HYPOTHESIS: Donchian(20) breakouts capture institutional accumulation/distribution. 
Filtered by 12h pivot direction (above/below weekly pivot) to align with higher timeframe bias.
Volume spike (>2x average) confirms participation. Works in bull/bear by following 12h trend.
Target: 75-150 total trades over 4 years (19-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1879_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for pivot and trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate weekly pivot from 12h data (approximate weekly from 12h)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # We'll approximate using rolling window on 12h (14 periods ≈ 1 week)
    window_1w = 14  # 14 * 12h = 168h = 1 week
    if len(high_12h) >= window_1w:
        weekly_high = pd.Series(high_12h).rolling(window=window_1w, min_periods=window_1w).max().values
        weekly_low = pd.Series(low_12h).rolling(window=window_1w, min_periods=window_1w).min().values
        weekly_close = pd.Series(close_12h).rolling(window=window_1w, min_periods=window_1w).mean().values  # approximate
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    else:
        weekly_pivot = np.full_like(close_12h, np.nan)
    
    # 12h trend: price above/below weekly pivot
    trend_12h = np.where(close_12h > weekly_pivot, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 6h Indicators: Donchian Channel(20) ===
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and weekly pivot
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Donchian reverse or adverse move ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below Donchian low (stop and reverse)
                if price < dc_low[i]:
                    exit_signal = True
                # Exit if 12h trend flips
                elif trend_12h_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above Donchian high (stop and reverse)
                if price > dc_high[i]:
                    exit_signal = True
                # Exit if 12h trend flips
                elif trend_12h_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h trend alignment for bias
        trend_bias = trend_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Donchian breakout in direction of 12h trend
            if trend_bias > 0 and price > dc_high[i]:  # Bullish breakout
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif trend_bias < 0 and price < dc_low[i]:  # Bearish breakdown
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals