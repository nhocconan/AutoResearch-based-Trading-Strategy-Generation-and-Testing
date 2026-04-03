#!/usr/bin/env python3
"""
Experiment #366: 4h Donchian(20) breakout + 1d volume confirmation + chop regime filter

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture institutional trends when 
confirmed by 1d volume spikes (>2.0x average) and traded only in trending regimes 
(chop < 61.8). Uses ATR-based trailing stoploss (2.5x ATR) and discrete position 
sizing (0.30) to minimize fee drag. Target: 75-200 total trades over 4 years on 4h.
Works in both bull/bear via symmetric long/short logic and regime filter preventing 
trades in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_vol_chop_v1"
timeframe = "4h"
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
    
    # === HTF: 1d data for chop regime filter (Call ONCE before loop) ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # +DM and -DM
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Choppiness Index: CHOP = 100 * log10(ATR_sum / (max_high - min_low)) / log10(n)
        atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
        chop = np.nan_to_num(chop, nan=50.0, posinf=100.0, neginf=0.0)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # Default to middle chop if insufficient data
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when CHOP < 61.8 (trending market) ---
        is_trending = chop_aligned[i] < 61.8
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Donchian(20) Breakout Calculation (using 4h data) ---
        lookback = 20
        if i >= lookback:
            highest_high = np.max(high[i-lookback+1:i+1])
            lowest_low = np.min(low[i-lookback+1:i+1])
            
            # Breakout conditions
            breakout_up = close[i] > highest_high
            breakout_down = close[i] < lowest_low
        else:
            highest_high = 0.0
            lowest_low = 0.0
            breakout_up = False
            breakout_down = False
        
        # --- Exit Logic (ATR-based trailing stop) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_level = highest_since_entry - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_level = lowest_since_entry + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + trending regime
        long_condition = breakout_up and volume_spike and is_trending
        
        # Short: Donchian breakout down + volume spike + trending regime
        short_condition = breakout_down and volume_spike and is_trending
        
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