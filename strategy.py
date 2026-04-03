#!/usr/bin/env python3
"""
Experiment #427: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, filtered by weekly pivot direction (price > weekly pivot = bullish bias for longs, < = bearish bias for shorts) and 1d volume confirmation (>1.5x average), creates a robust strategy that captures institutional breakouts in both bull and bear markets. Weekly pivots provide structural bias from higher timeframe, Donchian channels capture breakouts with clear stops, and volume confirms participation. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while focusing on high-probability breakouts aligned with weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Weekly pivot: P = (H+L+C)/3
        weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate Donchian(20) channels on 6h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            start_idx = i - lookback + 1
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
        # Values before lookback period remain NaN
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 20)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Weekly pivot direction for bias ---
        price_above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (Donchian-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                # Stoploss: break below Donchian low
                if low[i] < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit: break above Donchian high (trailing)
                if high[i] > highest_high[i]:
                    # Trail stop to break-even or better
                    entry_price = max(entry_price, lowest_low[i])  # Move stop to break-even
                signals[i] = SIZE
            else:  # Short position
                # Stoploss: break above Donchian high
                if high[i] > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit: break below Donchian low (trailing)
                if low[i] < lowest_low[i]:
                    # Trail stop to break-even or better
                    entry_price = min(entry_price, highest_high[i])  # Move stop to break-even
                signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian high with volume and weekly pivot bullish bias
        long_condition = (
            close[i] > highest_high[i] and  # Donchian breakout
            volume_spike and                # Volume confirmation
            price_above_weekly_pivot        # Weekly pivot bullish bias
        )
        
        # Short: Break below Donchian low with volume and weekly pivot bearish bias
        short_condition = (
            close[i] < lowest_low[i] and   # Donchian breakdown
            volume_spike and               # Volume confirmation
            price_below_weekly_pivot       # Weekly pivot bearish bias
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals