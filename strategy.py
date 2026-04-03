#!/usr/bin/env python3
"""
Experiment #132: 12h Donchian(20) Breakout + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: 12h Donchian breakouts confirmed by 1d volume spike (>2x average) and traded only in trending regimes (Choppiness Index < 38.2) capture intermediate-term momentum with low whipsaw. The 12h timeframe balances responsiveness with noise reduction, while the chop filter avoids ranging markets where breakouts fail. Discrete position sizing (0.25) and ATR trailing stop (2.0x) manage risk. Targets 12-37 trades/year to minimize fee drag and ensure statistical validity across BTC, ETH, and SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_1d_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume and chop regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d indicators
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # True Range for ATR and Chop
    tr_1d = np.zeros(len(close_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    atr_1d_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Choppiness Index (14)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14)
    # Handle division by zero or invalid
    chop = np.where((hh_14 - ll_14) > 0, chop, 50.0)  # neutral when range=0
    
    # Align 1d indicators to 12h
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- 1d Volume Confirmation (>2x average) ---
        vol_ok = volume[i] > vol_ma_20_aligned[i] * 2.0 if vol_ma_20_aligned[i] > 1e-10 else False
        
        # --- 1d Chop Regime Filter (< 38.2 = trending) ---
        chop_ok = chop_aligned[i] < 38.2
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: opposite Donchian touch or chop regime shift to ranging
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~1 day)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR chop becomes ranging
                    if close[i] <= dc_lower_20[i] or chop_aligned[i] >= 61.8:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR chop becomes ranging
                    if close[i] >= dc_upper_20[i] or chop_aligned[i] >= 61.8:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with volume confirmation and trending chop
        if bullish_breakout and vol_ok and chop_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and trending chop
        elif bearish_breakout and vol_ok and chop_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals