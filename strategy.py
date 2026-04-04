#!/usr/bin/env python3
"""
Experiment #4051: 6h Donchian(20) breakout + 1d HTF weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with weekly pivot bias (price above/below weekly pivot on 1d HTF) with volume confirmation capture high-probability continuation moves. Weekly pivot provides structural bias that works in both bull and bear markets by identifying key support/resistance levels from prior week. Volume confirmation filters false breakouts. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4051_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's 1d data
        # We need to aggregate 1d data into weekly OHLC
        # For simplicity, we'll use the last 5 trading days as proxy for week
        # In practice, we'd resample but we avoid resampling per rules
        # Instead, we calculate pivot points from prior 5-day period
        high_5d = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
        low_5d = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
        close_5d = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot: (H + L + C) / 3
        weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
        # Weekly R1: 2*P - L
        weekly_r1 = 2 * weekly_pivot - low_5d
        # Weekly S1: 2*P - H
        weekly_s1 = 2 * weekly_pivot - high_5d
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10)  # DC lookback, vol MA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Weekly pivot bias: price above pivot for long bias, below for short bias
            price_above_pivot = price > weekly_pivot_aligned[i]
            price_below_pivot = price < weekly_pivot_aligned[i]
            
            # Breakout logic: 
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long conditions: above weekly pivot + upper Donchian breakout
            long_entry = breakout_up and price_above_pivot
            
            # Short conditions: below weekly pivot + lower Donchian breakout
            short_entry = breakout_down and price_below_pivot
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals