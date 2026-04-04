#!/usr/bin/env python3
"""
Experiment #4967: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) with volume confirmation (>1.5x average) capture strong momentum moves in both bull and bear markets. Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4967_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data
        # We'll use rolling window of 5 days to approximate weekly OHLC
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    else:
        weekly_pivot = np.full(len(df_1d), np.nan)
    
    # Align HTF weekly pivot to 6h timeframe
    if len(weekly_pivot) > 0:
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20)  # Donchian, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal ---
        if in_position:
            # Check for reverse signal
            vol_confirm = vol_ratio[i] > 1.5
            
            # Long exit conditions
            if position_side > 0:  # Long
                reverse_signal = (price <= low_roll[i]) and (price < weekly_pivot_aligned[i]) and vol_confirm
                if reverse_signal:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            # Short exit conditions
            else:  # Short
                reverse_signal = (price >= high_roll[i]) and (price > weekly_pivot_aligned[i]) and vol_confirm
                if reverse_signal:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with weekly pivot alignment
        breakout_long = (price >= high_roll[i]) and (price > weekly_pivot_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < weekly_pivot_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals