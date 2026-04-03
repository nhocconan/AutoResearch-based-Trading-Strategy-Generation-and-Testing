#!/usr/bin/env python3
"""
Experiment #451: 6h Williams %R + 1d Supertrend + Volume Filter

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe, 
while 1d Supertrend provides the higher timeframe trend direction. Volume confirmation 
ensures institutional participation. This combination works in both bull and bear markets 
by taking mean reversion trades at extremes during pullbacks in the prevailing trend. 
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) with discrete 
position sizing to minimize fee drag.
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
        
        # ATR calculation
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
        
        # Supertrend calculation
        hl2 = (high_1d + low_1d) / 2
        upperband = hl2 + (3 * atr)
        lowerband = hl2 - (3 * atr)
        
        supertrend = np.full(len(close_1d), np.nan)
        direction = np.full(len(close_1d), np.nan)  # 1 for uptrend, -1 for downtrend
        
        supertrend[0] = upperband[0]
        direction[0] = 1
        
        for i in range(1, len(close_1d)):
            if close_1d[i] > supertrend[i-1]:
                supertrend[i] = max(lowerband[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(upperband[i], supertrend[i-1])
                direction[i] = -1
        
        # Align to 6h timeframe
        supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
        direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    else:
        supertrend_aligned = np.full(n, np.nan)
        direction_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Williams %R(14)
    williams_r = np.full(n, np.nan)
    if n >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero
        williams_r[highest_high == lowest_low] = -50
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = np.zeros(n)
    volume_spike[20:] = volume[20:] / vol_ma_20[20:]
    volume_spike[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1d Supertrend ---
        is_uptrend = direction_aligned[i] > 0
        is_downtrend = direction_aligned[i] < 0
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_confirmed = volume_spike[i] > 1.5
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Exit when Williams %R returns to neutral territory
            if (position_side > 0 and williams_r[i] > -50) or \
               (position_side < 0 and williams_r[i] < -50):
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R oversold (< -80) in uptrend with volume confirmation
        long_condition = (
            williams_r[i] < -80 and 
            is_uptrend and 
            volume_confirmed
        )
        
        # Short: Williams %R overbought (> -20) in downtrend with volume confirmation
        short_condition = (
            williams_r[i] > -20 and 
            is_downtrend and 
            volume_confirmed
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