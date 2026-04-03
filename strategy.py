#!/usr/bin/env python3
"""
Experiment #391: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, filtered by weekly pivot direction (from 1w timeframe) and 
confirmed by volume spike on 1d, creates a robust trend-following strategy that works in both bull and bear markets. 
Weekly pivot provides institutional reference points for trend direction, Donchian captures breakouts, and volume 
confirms institutional participation. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize 
fee drag while capturing high-probability breakouts aligned with weekly structure.
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
    
    # === HTF: 1w data for weekly pivot direction (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H)
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        pivot_point = np.zeros(len(df_1w))
        r1 = np.zeros(len(df_1w))
        s1 = np.zeros(len(df_1w))
        
        for i in range(len(df_1w)):
            pivot_point[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
            r1[i] = 2 * pivot_point[i] - low_1w[i]
            s1[i] = 2 * pivot_point[i] - high_1w[i]
        
        pivot_point_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
        r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    else:
        pivot_point_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate Donchian Channel (20-period) on 6h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            start_idx = i - lookback + 1
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
        else:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_point_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Weekly pivot direction (price > pivot = uptrend, < pivot = downtrend) ---
        price_above_pivot = close[i] > pivot_point_aligned[i]
        price_below_pivot = close[i] < pivot_point_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly R1 (strong resistance) or S1 (strong support)
                if close[i] >= r1_aligned[i] or close[i] <= s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly R1 (strong resistance) or S1 (strong support)
                if close[i] >= r1_aligned[i] or close[i] <= s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above highest_high with weekly uptrend and volume
        long_condition = (
            close[i] > highest_high[i] and  # Donchian breakout
            price_above_pivot and           # Weekly uptrend filter
            volume_spike                    # Volume confirmation
        )
        
        # Short: Donchian breakdown below lowest_low with weekly downtrend and volume
        short_condition = (
            close[i] < lowest_low[i] and    # Donchian breakdown
            price_below_pivot and           # Weekly downtrend filter
            volume_spike                    # Volume confirmation
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