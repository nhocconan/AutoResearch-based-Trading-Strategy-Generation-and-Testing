#!/usr/bin/env python3
"""
Experiment #003: 4h Donchian(20) Breakout + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by volume spikes (>2.0x average),
capture strong momentum moves in both bull and bear markets. The Donchian structure provides objective
breakout levels, volume confirmation filters out false breakouts, and ATR-based stoploss manages risk.
Target 75-200 trades over 4 years (19-50/year) to minimize fee drag while capturing high-probability trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 20  # Ensure enough data for Donchian and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike
        long_condition = breakout_up and volume_spike
        
        # Short: Donchian breakout down + volume spike
        short_condition = breakout_down and volume_spike
        
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