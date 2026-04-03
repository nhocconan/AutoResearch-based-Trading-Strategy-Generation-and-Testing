#!/usr/bin/env python3
"""
Experiment #319: 6h Williams %R + 12h Supertrend + Volume Confirmation

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe, 
while 12h Supertrend provides the higher timeframe trend direction. Volume confirmation 
ensures institutional participation. This combination works in both bull and bear markets 
by taking mean-reversion entries at extremes when aligned with HTF trend, targeting 
12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_12h_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Supertrend on 12h
    if len(df_12h) >= 10:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # ATR calculation
        tr1 = high_12h - low_12h
        tr2 = np.abs(high_12h - np.roll(close_12h, 1))
        tr3 = np.abs(low_12h - np.roll(close_12h, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        atr = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
        
        # Supertrend calculation
        hl2 = (high_12h + low_12h) / 2
        upperband = hl2 + (3 * atr)
        lowerband = hl2 - (3 * atr)
        
        supertrend = np.zeros(len(close_12h))
        direction = np.ones(len(close_12h))  # 1 for uptrend, -1 for downtrend
        
        supertrend[0] = upperband[0]
        direction[0] = 1
        
        for i in range(1, len(close_12h)):
            if close_12h[i] > supertrend[i-1]:
                direction[i] = 1
            elif close_12h[i] < supertrend[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = max(lowerband[i], supertrend[i-1])
            else:
                supertrend[i] = min(upperband[i], supertrend[i-1])
        
        # Align to 6h timeframe
        supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
        direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    else:
        supertrend_aligned = np.full(n, np.nan)
        direction_aligned = np.full(n, np.nan)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Williams %R(14)
    williams_r = np.full(n, np.nan)
    if n >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero
        williams_r[highest_high == lowest_low] = -50
    
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
            np.isnan(direction_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Use 12h Supertrend direction ---
        trend_up = direction_aligned[i] > 0
        trend_down = direction_aligned[i] < 0
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.5
        
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
                # Take profit at Williams %R extremes
                if williams_r[i] >= -20:  # Overbought
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
                # Take profit at Williams %R extremes
                if williams_r[i] <= -80:  # Oversold
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R oversold (-80 to -100) in uptrend with volume
        long_condition = (
            williams_r[i] <= -80 and  # Oversold
            trend_up and              # HTF uptrend
            volume_spike              # Volume confirmation
        )
        
        # Short: Williams %R overbought (-20 to 0) in downtrend with volume
        short_condition = (
            williams_r[i] >= -20 and  # Overbought
            trend_down and            # HTF downtrend
            volume_spike              # Volume confirmation
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