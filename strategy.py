#!/usr/bin/env python3
"""
Experiment #042: 12h Donchian Breakout + Volume Spike + Chop Regime Filter

HYPOTHESIS: 12h Donchian(20) breakouts capture significant momentum moves. 
Combined with 1d volume confirmation (>2x average) and 1w choppiness regime 
(CHOP > 61.8 = ranging, < 38.2 = trending), this strategy filters false breakouts 
in choppy markets while capturing strong trends. Uses discrete sizing (0.25) 
and ATR-based stoploss. Targets 50-150 trades over 4 years on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.ones(len(vol_1d))  # Default to 1.0 (neutral)
        if len(vol_1d) > 20:
            vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for choppiness regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index(14) on 1w
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = np.zeros(len(high_1w))
        tr1[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(high_1w)):
            tr1[i] = max(high_1w[i] - low_1w[i], 
                         abs(high_1w[i] - close_1w[i-1]), 
                         abs(low_1w[i] - close_1w[i-1]))
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index
        chop = np.full(len(high_1w), 50.0)  # Default neutral
        for i in range(14, len(high_1w)):
            if hh_14[i] != ll_14[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
            else:
                chop[i] = 50.0
        
        chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # Neutral if insufficient data
    
    # === Primary: 12h Donchian(20) breakout ===
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    if n >= lookback:
        for i in range(lookback - 1, n):
            window_high = high[i - lookback + 1:i + 1]
            window_low = low[i - lookback + 1:i + 1]
            highest_high[i] = np.max(window_high)
            lowest_low[i] = np.min(window_low)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(100, lookback)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when market is trending (CHOP < 38.2) ---
        is_trending = chop_aligned[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 2x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
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
                # Update highest since entry for trailing stop logic
                highest_since_entry = max(highest_since_entry, high[i])
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Update lowest since entry for trailing stop logic
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Donchian breakout: Long when price > 20-period high, Short when price < 20-period low
        long_condition = (
            close[i] > highest_high[i] and 
            volume_spike and 
            is_trending
        )
        
        short_condition = (
            close[i] < lowest_low[i] and 
            volume_spike and 
            is_trending
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals