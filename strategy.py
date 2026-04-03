#!/usr/bin/env python3
"""
Experiment #371: 6h Williams %R + 1d Supertrend + Volume Confirmation

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe, 
combined with 1d Supertrend for trend direction and volume spike confirmation (>2.0x average) 
to filter false signals. This strategy aims to capture mean reversions in ranging markets 
and trend continuations in strong trends, targeting 12-37 trades/year (50-150 total over 4 years) 
to minimize fee drag while maintaining statistical validity. Works in both bull and bear markets 
by using Supertrend to adapt to prevailing trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_1d_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Supertrend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Supertrend on 1d
    if len(df_1d) >= 10:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # ATR(10)
        atr = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
        
        # Supertrend calculation
        hl2 = (high_1d + low_1d) / 2
        upperband = hl2 + (3.0 * atr)
        lowerband = hl2 - (3.0 * atr)
        
        supertrend = np.zeros(len(close_1d))
        direction = np.ones(len(close_1d))  # 1 for uptrend, -1 for downtrend
        
        supertrend[0] = upperband[0]
        direction[0] = 1
        
        for i in range(1, len(close_1d)):
            if close_1d[i] > supertrend[i-1]:
                direction[i] = 1
            elif close_1d[i] < supertrend[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1 and direction[i-1] == -1:
                supertrend[i] = upperband[i]
            elif direction[i] == -1 and direction[i-1] == 1:
                supertrend[i] = lowerband[i]
            elif direction[i] == 1:
                supertrend[i] = max(upperband[i], supertrend[i-1])
            else:
                supertrend[i] = min(lowerband[i], supertrend[i-1])
        
        # Align Supertrend and direction to 6h timeframe
        supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
        direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    else:
        supertrend_aligned = np.full(n, np.nan)
        direction_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume average (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Williams %R(14)
    williams_r = np.full(n, np.nan)
    if n >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero
        williams_r[highest_high == lowest_low] = -50.0
    
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
        if (np.isnan(williams_r[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss on 6h
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Williams %R levels: > -20 = overbought, < -80 = oversold
        # In uptrend (direction=1): look for oversold reversals (< -80)
        # In downtrend (direction=-1): look for overbought reversals (> -20)
        long_condition = (
            (williams_r[i] < -80.0) and  # Oversold
            (direction_aligned[i] > 0) and  # Uptrend on 1d
            volume_spike  # Volume confirmation
        )
        
        short_condition = (
            (williams_r[i] > -20.0) and  # Overbought
            (direction_aligned[i] < 0) and  # Downtrend on 1d
            volume_spike  # Volume confirmation
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