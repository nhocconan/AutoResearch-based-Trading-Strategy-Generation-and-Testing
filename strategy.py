#!/usr/bin/env python3
"""
Experiment #5107: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) capture strong momentum with institutional reference points. Volume > 1.5x average confirms participation. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Weekly pivot provides structural bias that works in both bull (buy breakouts above pivot) and bear (sell breakdowns below pivot) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5107_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Weekly Pivot Point (using prior week's OHLC) ===
    if len(df_1d) >= 5:  # Need at least 5 days for prior week
        # Calculate weekly OHLC from daily data
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1)
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1)
        
        # Weekly Pivot Point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pivot_values = weekly_pivot.values
    else:
        weekly_pivot_values = np.full(len(df_1d), np.nan)
    
    # Align weekly pivot to 6h timeframe (shifted by 1 for completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_values)
    
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
            
            # Long exit: price breaks below weekly pivot OR Donchian breakdown with volume
            exit_long = (price < weekly_pivot_aligned[i]) or \
                       ((price <= low_roll[i]) and vol_confirm)
            
            # Short exit: price breaks above weekly pivot OR Donchian breakout with volume
            exit_short = (price > weekly_pivot_aligned[i]) or \
                        ((price >= high_roll[i]) and vol_confirm)
            
            if (position_side > 0 and exit_long) or (position_side < 0 and exit_short):
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with weekly pivot bias
        # Long: Donchian breakout above + price > weekly pivot (bullish bias)
        # Short: Donchian breakdown below + price < weekly pivot (bearish bias)
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