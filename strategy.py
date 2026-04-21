#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Confluence_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) and volume confirmation (volume > 1.5x 20-period average) captures high-probability trend continuation moves. Weekly pivot acts as institutional reference point reducing false breakouts. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly Pivot (standard calculation) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # === 6h Donchian(20) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 2 days (8 * 6h = 48h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        weekly_pivot = pivot_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly pivot bias
        is_bull_bias = price > weekly_pivot
        is_bear_bias = price < weekly_pivot
        
        if position == 0:
            # Long: price breaks above Donchian upper + bull bias + volume
            long_condition = (price > dc_upper[i]) and is_bull_bias and vol_conf
            # Short: price breaks below Donchian lower + bear bias + volume
            short_condition = (price < dc_lower[i]) and is_bear_bias and vol_conf
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR approximation using Donchian width)
            donchian_width = dc_upper[i] - dc_lower[i]
            atr_approx = donchian_width * 0.2  # rough approximation
            
            if position == 1:
                if price < entry_price - 2.5 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_approx:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Confluence_v1"
timeframe = "6h"
leverage = 1.0