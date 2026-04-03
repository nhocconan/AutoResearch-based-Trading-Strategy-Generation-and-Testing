#!/usr/bin/env python3
"""
Experiment #1871: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts capture strong momentum moves. Weekly pivot direction (from 1d HTF) filters for breakouts aligned with the higher-timeframe trend. Volume confirmation (>1.5x average) ensures breakouts have institutional participation. Works in both bull and bear markets by following the 1d trend via weekly pivot bias. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1871_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week's 1d data
    # We need to group 1d data into weeks (Mon-Sun) and calculate pivot for prior week
    # Simplified: use rolling window of 5 trading days (1 week) to approximate weekly pivot
    if len(close_1d) >= 5:
        # Weekly high/low/close from prior 5-day window (shifted by 1 to avoid look-ahead)
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
        
        # Weekly pivot point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Bias: price above weekly pivot = bullish bias, below = bearish bias
        weekly_bias = np.where(close_1d > weekly_pivot, 1, -1)
        # Align to 6h timeframe
        weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    else:
        weekly_bias_aligned = np.ones(n)  # neutral bias if insufficient data
    
    # === 6h Indicators: Donchian Channel (20) ===
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
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or stoploss ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below Donchian low (failed breakout)
                if price < dc_low[i]:
                    exit_signal = True
                # Exit if weekly bias flips to bearish
                elif weekly_bias_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above Donchian high (failed breakout)
                if price > dc_high[i]:
                    exit_signal = True
                # Exit if weekly bias flips to bullish
                elif weekly_bias_aligned[i] > 0:
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
        # Require weekly pivot bias for direction
        bias = weekly_bias_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long breakout: price breaks above Donchian high with bullish bias
            if bias > 0 and price > dc_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short breakout: price breaks below Donchian low with bearish bias
            elif bias < 0 and price < dc_low[i]:
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