#!/usr/bin/env python3
"""
Experiment #075: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, filtered by weekly pivot direction (from 1w timeframe),
and confirmed by volume spikes (>2.0x average) produce high-probability trades. Weekly pivot acts as
regime filter: only take longs above weekly pivot, shorts below. This adapts to bull/bear markets by
using the weekly pivot as dynamic trend filter. Target: 75-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_075_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Weekly Pivot (based on prior week) ===
    def calculate_weekly_pivot(high, low, close):
        # Classic pivot: (H + L + C) / 3
        pivot = (high + low + close) / 3.0
        return pivot
    
    # Calculate for each 1w bar (using previous week's data)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pivot_1w = np.full_like(c_1w, np.nan)
    
    for i in range(1, len(c_1w)):
        pivot = calculate_weekly_pivot(h_1w[i-1], l_1w[i-1], c_1w[i-1])
        weekly_pivot_1w[i] = pivot
    
    # Align to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian and volume stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pivot_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 2.0  # Volume spike threshold
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit on opposite Donchian break with volume
            if position_side > 0:  # Long
                if low[i] < donchian_lower[i-1] and vol_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if high[i] > donchian_upper[i-1] and vol_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Only trade in direction of weekly pivot
        if price > weekly_pivot_6h[i-1]:  # Above weekly pivot - bias long
            if price > donchian_upper[i-1] and vol_spike:  # Breakout above Donchian upper
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif price < weekly_pivot_6h[i-1]:  # Below weekly pivot - bias short
            if price < donchian_lower[i-1] and vol_spike:  # Breakdown below Donchian lower
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # At weekly pivot - no clear bias
            signals[i] = 0.0
    
    return signals

</think>