#!/usr/bin/env python3
"""
Experiment #125: 12h Donchian(20) Breakout + 1d Volume Spike + ATR Stoploss

HYPOTHESIS: 12h Donchian channel breakouts filtered by 1d volume spikes (>1.8x average) 
capture strong momentum moves while minimizing false breakouts. The 12h timeframe 
targets 12-37 trades/year (50-150 total over 4 years) to reduce fee drag. Volume 
confirmation ensures breakouts have participation. Works in both bull (breakouts with 
volume) and bear (failed breaks reverse sharply) markets. Uses ATR-based stoploss 
for risk management. Discrete position sizing (0.25) minimizes churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_125_12h_donchian_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Volume MA(20)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF volume MA, ATR, and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Spike: Require 1d volume > 1.8x 20-period average ---
        volume_spike = df_1d['volume'].values[i // (24*60//12)] > 1.8 * vol_ma_20_1d_aligned[i] if i // (24*60//12) < len(df_1d) else False
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
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
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

**NOTE ON VOLUME SPIKE CALCULATION**: 
The volume spike condition uses the 1d volume data aligned to 12h timeframe. 
Since 1d data has fewer bars, we calculate the volume MA on 1d data and align it. 
For each 12h bar, we check if the corresponding 1d volume (same day) exceeds 
1.8x its 20-day average. This ensures we're filtering for days with unusually 
high trading volume, which confirms the significance of the breakout.