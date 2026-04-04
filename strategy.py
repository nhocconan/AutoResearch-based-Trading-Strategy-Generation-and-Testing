#!/usr/bin/env python3
"""
Experiment #3627: 6h Donchian(20) breakout + 1d weekly pivot direction + volume spike
HYPOTHESIS: 6h Donchian breakouts capture momentum bursts. Weekly pivot levels from 1d provide directional bias (price above weekly pivot = bullish bias, below = bearish bias). Volume spike confirms breakout validity. Works in bull markets (breakouts continue uptrend) and bear markets (breakdowns continue downtrend). Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3627_6h_donchian20_1d_pivot_vol_v1"
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
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We'll use rolling window of 5 days (1 week) to get prior week's values
    lookback_week = 5
    if len(high_1d) >= lookback_week:
        # Shift by 1 to use prior week's data (avoid look-ahead)
        week_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().shift(1).values
        week_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().shift(1).values
        week_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).mean().shift(1).values
        weekly_pivot = (week_high + week_low + week_close) / 3.0
    else:
        weekly_pivot = np.full(len(close_1d), np.nan)
    
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_donchian = 20
    highest_high = pd.Series(high).rolling(window=lookback_donchian, min_periods=lookback_donchian).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_donchian, min_periods=lookback_donchian).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, lookback_donchian + 1, lookback_week + 1, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price moves 2.5*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine bias from weekly pivot: price above pivot = bullish, below = bearish
            bullish_bias = price > weekly_pivot_aligned[i]
            
            # Long entry: price breaks above Donchian high + bullish bias + volume spike
            if (price > highest_high[i] and 
                bullish_bias):
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low + bearish bias + volume spike
            elif (price < lowest_low[i] and 
                  not bullish_bias):
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals