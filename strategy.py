#!/usr/bin/env python3
"""
Experiment #3811: 6h Donchian(20) breakout + 1d volume spike + weekly pivot direction
HYPOTHESIS: 6h Donchian breakouts capture medium-term swings with 1d volume (>2.0x) confirming institutional participation. Weekly pivot (from 1d HTF) provides directional bias: only long when price above weekly pivot, only short when below. This avoids counter-trend trades in strong weekly trends. Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3811_6h_donchian20_1d_vol_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from prior week's OHLC (using 1d data)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    lookback_week = 5  # 5 trading days per week
    weekly_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().values
    weekly_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().values
    weekly_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
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
    
    warmup = max(lookback_dc + 1, 20, lookback_week)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price breaks below Donchian lower band (long) or above upper band (short)
            if position_side > 0:  # Long
                if price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for institutional confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above weekly pivot (bullish alignment)
            if (price > highest_high[i-1] and    # Breakout above previous period's high
                price > weekly_pivot_aligned[i]):  # Above weekly pivot (bullish bias)
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below weekly pivot (bearish alignment)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < weekly_pivot_aligned[i]):  # Below weekly pivot (bearish bias)
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals