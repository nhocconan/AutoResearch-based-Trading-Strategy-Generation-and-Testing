#!/usr/bin/env python3
"""
Experiment #947: 6h Donchian Breakout + Weekly Pivot + Volume Confirmation
HYPOTHESIS: Combining 6h Donchian(20) breakouts with weekly pivot point direction 
filters trades to align with higher timeframe structure. Volume confirmation (>1.8x avg) 
ensures institutional participation. This reduces false breakouts in ranging markets 
while capturing strong trends. Weekly pivot provides bias: only long when price > weekly 
pivot, only short when price < weekly pivot. Target: 75-150 total trades over 4 years 
(19-37/year) on 6h timeframe. Works in both bull (breakouts) and bear (breakdowns) 
markets by following HTF pivot direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_947_6h_donchian_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Donchian channels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 1d: 20-period high/low
    # We'll use pandas rolling for clarity, then convert to numpy
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donch_high_1d = high_1d_series.rolling(window=20, min_periods=20).max().values
    donch_low_1d = low_1d_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # === HTF: 1w data for weekly pivot point (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
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
    
    warmup = max(20, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
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
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Breakout continuation: price breaks above Donchian high OR below Donchian low
            # BUT only in direction of weekly pivot bias
            if price > donch_high_aligned[i] and price > weekly_pivot_aligned[i]:
                # Bullish breakout above weekly pivot bias
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low_aligned[i] and price < weekly_pivot_aligned[i]:
                # Bearish breakdown below weekly pivot bias
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