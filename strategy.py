#!/usr/bin/env python3
"""
Experiment #307: 6h Donchian(20) breakout + 1d Weekly Pivot direction + volume confirmation
HYPOTHESIS: Price breaking 6h Donchian(20) channels with 1d Weekly Pivot (R4/S4) trend filter and volume confirmation (>1.5x) captures strong momentum moves aligned with weekly structure. Weekly pivot provides robust support/resistance levels that work in both bull (breakout continuation) and bear (mean reversion at extremes) markets. Uses discrete sizing (0.25) to minimize fee drag. Target: 75-150 total trades over 4 years for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_307_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Weekly Pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Weekly Pivot levels from prior week's OHLC
    # We need to group 1d data by week and calculate pivot for each week
    # Since we don't have direct weekly data, we'll approximate using rolling weekly lookback
    # Weekly Pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # R4 = Prior Week Close + 3*(Prior Week High - Prior Week Low)
    # S4 = Prior Week Close - 3*(Prior Week High - Prior Week Low)
    
    # Calculate rolling weekly (5-day) high, low, close for approximation
    # Using 5-day roll to approximate prior week (since 1d data)
    week_high = pd.Series(high).rolling(window=5, min_periods=5).max().shift(1).values  # Prior week high
    week_low = pd.Series(low).rolling(window=5, min_periods=5).min().shift(1).values    # Prior week low
    week_close = pd.Series(close).rolling(window=5, min_periods=5).last().shift(1).values  # Prior week close
    
    # Weekly Pivot Point
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    # Weekly R4 and S4 levels
    weekly_range = week_high - week_low
    weekly_r4 = week_close + 3.0 * weekly_range
    weekly_s4 = week_close - 3.0 * weekly_range
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + 5-day weekly lookback
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Weekly Pivot Trend Filter ---
        # Uptrend bias: price above weekly pivot AND approaching R4
        # Downtrend bias: price below weekly pivot AND approaching S4
        # We use the pivot as midline, R4/S4 as extreme levels
        above_pivot = price > weekly_pivot_aligned[i]
        below_pivot = price < weekly_pivot_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss (using 2.0*ATR for standard stops) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr[0] = high[0] - low[0]
                atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_val = 0.0
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 20 bars (5 days on 6h)
            if bars_since_entry > 20:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND price above weekly pivot (uptrend bias)
            if breakout_up and above_pivot:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND price below weekly pivot (downtrend bias)
            elif breakout_down and below_pivot:
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