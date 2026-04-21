#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1
Hypothesis: Donchian(20) breakouts on 6h filtered by weekly pivot direction (from 1w HTF) and 6h volume confirmation.
Weekly pivot direction provides multi-week structural bias: above weekly pivot = bull bias (favor longs), below = bear bias (favor shorts).
Volume confirmation ensures institutional participation. Discrete sizing (0.25) targets 12-37 trades/year on 6h.
Uses 1w HTF for pivot calculation to avoid look-ahead and capture true weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for ATR/volume, 1w for weekly pivot)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1d ATR (14-period) for stoploss ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d volume confirmation (volume > 1.8x 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > (1.8 * vol_ma_20_1d)
    volume_confirmed_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d)
    
    # === 1w weekly pivot (standard calculation) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly pivot direction: 1 = above pivot (bull bias), -1 = below pivot (bear bias)
    pivot_direction_1w = np.where(close_1w > pivot_1w, 1, -1)
    pivot_direction_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_direction_1w)
    
    # === 6h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower (20-period lookback)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(pivot_direction_1w_aligned[i]) or 
            np.isnan(volume_confirmed_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        atr_val = atr_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        pivot_dir = pivot_direction_1w_aligned[i]
        vol_conf = volume_confirmed_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper channel WITH bull bias from weekly pivot
            long_condition = (price > upper) and vol_conf and (pivot_dir == 1)
            # Short breakdown: price < lower channel WITH bear bias from weekly pivot
            short_condition = (price < lower) and vol_conf and (pivot_dir == -1)
            
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
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below lower channel (failed breakout)
                elif price < lower:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks above upper channel (failed breakdown)
                elif price > upper:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0