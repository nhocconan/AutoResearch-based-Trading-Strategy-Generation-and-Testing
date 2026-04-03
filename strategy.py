#!/usr/bin/env python3
"""
Experiment #219: 6h Donchian Breakout with Weekly Pivot Filter

HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1w HTF) 
capture strong trending moves while avoiding counter-trend noise. Weekly pivot acts as 
institutional reference point: breakouts above weekly R1 in uptrend (price > weekly PP) 
or below weekly S1 in downtrend (price < weekly PP) have higher follow-through. 
Volume confirmation filters false breakouts. 6h timeframe targets 12-37 trades/year 
(50-150 total over 4 years) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_219_6h_donchian_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    def calculate_pivot(h, l, c):
        """Calculate standard pivot points: R2, R1, PP, S1, S2"""
        pp = (h + l + c) / 3.0
        r1 = 2 * pp - l
        s1 = 2 * pp - h
        r2 = pp + (h - l)
        s2 = pp - (h - l)
        return r2, r1, pp, s1, s2
    
    # Calculate for each 1w bar
    r2_1w = np.full(len(df_1w), np.nan)
    r1_1w = np.full(len(df_1w), np.nan)
    pp_1w = np.full(len(df_1w), np.nan)
    s1_1w = np.full(len(df_1w), np.nan)
    s2_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        r2, r1, pp, s1, s2 = calculate_pivot(
            df_1w['high'].iloc[i],
            df_1w['low'].iloc[i],
            df_1w['close'].iloc[i]
        )
        r2_1w[i] = r2
        r1_1w[i] = r1
        pp_1w[i] = pp
        s1_1w[i] = s1
        s2_1w[i] = s2
    
    # Align weekly pivot levels to 6h timeframe
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period):
        """Calculate Donchian Channel upper and lower bands"""
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for Donchian and volume indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(r1_1w_aligned[i]) or np.isnan(pp_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        r1_w = r1_1w_aligned[i]
        pp_w = pp_1w_aligned[i]
        s1_w = s1_1w_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: Require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Long breakout: Price > Donchian upper + volume spike + price > weekly PP (uptrend bias)
        long_breakout = (price > upper) and volume_spike and (price > pp_w)
        
        # Short breakout: Price < Donchian lower + volume spike + price < weekly PP (downtrend bias)
        short_breakout = (price < lower) and volume_spike and (price < pp_w)
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals