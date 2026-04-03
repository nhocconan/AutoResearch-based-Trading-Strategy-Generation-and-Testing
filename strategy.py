#!/usr/bin/env python3
"""
Experiment #951: 6h Donchian(20) Breakout + 1d Weekly Pivot + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 6h timeframe capture momentum, filtered by weekly pivot direction from 1d timeframe to avoid counter-trend trades. Volume spike (>2.0x 20-period average) confirms institutional participation. Long when price breaks above Donchian(20) high AND weekly pivot > previous weekly pivot (bullish bias). Short when price breaks below Donchian(20) low AND weekly pivot < previous weekly pivot (bearish bias). Uses discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) to manage drawdown. Target: 75-150 total trades over 4 years (19-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_951_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from daily data (using last 5 trading days)
    # Weekly pivot = (Weekly High + Weekly Low + Weekly Close) / 3
    # We approximate using rolling window of 5 days on 1d data
    window = 5
    if len(high_1d) >= window:
        # Rolling weekly high/low/close
        weekly_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
        weekly_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
        weekly_close = pd.Series(close_1d).rolling(window=window, min_periods=window).last().values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Previous weekly pivot for bias
        weekly_pivot_prev = np.roll(weekly_pivot, 1)
        weekly_pivot_prev[0] = weekly_pivot_prev[1] if len(weekly_pivot) > 1 else weekly_pivot[0]
    else:
        # Not enough data, use daily pivot as fallback
        pivot_1d = (high_1d + low_1d + close_1d) / 3.0
        weekly_pivot = pivot_1d
        weekly_pivot_prev = np.roll(pivot_1d, 1)
        weekly_pivot_prev[0] = weekly_pivot_prev[1] if len(pivot_1d) > 1 else pivot_1d[0]
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_pivot_prev_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_prev)
    
    # === 6h Indicators: Donchian(20) channels ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(donchian_window, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_pivot_prev_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~3d on 6h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine weekly pivot bias: bullish if current > previous, bearish if current < previous
            weekly_bullish = weekly_pivot_aligned[i] > weekly_pivot_prev_aligned[i]
            weekly_bearish = weekly_pivot_aligned[i] < weekly_pivot_prev_aligned[i]
            
            # Breakout continuation: price breaks above Donchian high with bullish bias OR below Donchian low with bearish bias
            if price > donchian_high[i] and weekly_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donchian_low[i] and weekly_bearish:
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