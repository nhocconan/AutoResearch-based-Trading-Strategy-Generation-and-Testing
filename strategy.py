#!/usr/bin/env python3
"""
Experiment #186: 4h Donchian(20) breakout + 1d volume spike + ATR trailing stop
HYPOTHESIS: Donchian breakouts on 4h aligned with 1d volume spikes capture institutional participation.
Uses ATR(14) trailing stop (highest high since entry - 3*ATR for longs, lowest low + 3*ATR for shorts).
No HTF indicators needed - pure price/volume action. Discrete sizing 0.25 minimizes fee churn.
Target: 100-180 total trades over 4 years (25-45/year). Works in bull (breakouts continue) and bear (failed breaks reverse sharply).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_186_4h_donchian20_1d_vol_spike_atr_stop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume MA(20) for spike detection
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.zeros(len(df_1d))
    vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_1d[20:]
    vol_ratio_1d[:20] = 1.0
    
    # Align 1d volume ratio to 4h timeframe
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    highest_since_entry = 0.0  # for trailing stop (longs)
    lowest_since_entry = 0.0   # for trailing stop (shorts)
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require 1d volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Exit Logic (ATR-based trailing stop) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Update highest high since entry
                if high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                # Trailing stop: highest high - 3*ATR
                stop_level = highest_since_entry - 3.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    highest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian break with volume confirmation
                if breakout_down and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    highest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update lowest low since entry
                if low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                # Trailing stop: lowest low + 3*ATR
                stop_level = lowest_since_entry + 3.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    lowest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Donchian break with volume confirmation
                if breakout_up and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    lowest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + Donchian breakout
        if volume_spike:
            if breakout_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif breakout_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals