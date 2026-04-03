#!/usr/bin/env python3
"""
Experiment #071: 6h Williams %R Mean Reversion + 1d Volume Spike + ATR Stop

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe.
In ranging markets (2021-2024), price reverts from extremes with 1d volume confirmation
providing institutional participation signal. ATR-based stops manage risk during trends.
Discrete sizing (0.25) reduces fee churn. Target: 15-30 trades/year for statistical
validity with minimal fee drag. Works in both bull/bear via mean reversion logic.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d volume MA for confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 6h Indicators ===
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    atr_14 = pd.Series(high - low).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 20  # Ensure enough data for Williams %R and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
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
            
            # Exit conditions: Williams %R returns to neutral zone (between -20 and -80)
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: Williams %R rises above -20 (overbought)
                    if williams_r[i] > -20:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: Williams %R falls below -80 (oversold)
                    if williams_r[i] < -80:
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
        # Volume confirmation: current 6h volume > 1.5x 1d average volume (scaled)
        # 1d volume MA represents average daily volume; 6h is 1/4 of day
        vol_6h_expected = vol_ma_20_1d_aligned[i] / 4.0
        vol_ok = volume[i] > vol_6h_expected * 1.5 if vol_6h_expected > 1e-10 else False
        
        # Long conditions: 
        # Williams %R below -80 (oversold) with volume confirmation
        if williams_r[i] < -80 and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Williams %R above -20 (overbought) with volume confirmation
        elif williams_r[i] > -20 and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals