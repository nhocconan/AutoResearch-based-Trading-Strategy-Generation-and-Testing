#!/usr/bin/env python3
"""
Experiment #4995: 6h Donchian(20) Breakout + 1w Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts in direction of weekly pivot bias (price > weekly pivot = long bias, price < weekly pivot = short bias) with volume confirmation (>1.5x average) capture strong momentum moves while minimizing whipsaw. Weekly pivot provides structural bias from higher timeframe, reducing counter-trend entries. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with bullish bias) and bear markets (breakdowns with bearish bias).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4995_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for weekly pivot bias
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Weekly Pivot (standard floor pivot) ===
    if len(df_1w) >= 1:
        # Weekly pivot: P = (H + L + C) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        pivot_1w = (weekly_high + weekly_low + weekly_close) / 3.0
    else:
        pivot_1w = np.array([])
    
    # Align HTF weekly pivot to 6h timeframe
    if len(pivot_1w) > 0:
        pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    else:
        pivot_1w_aligned = np.full(n, np.nan)
    
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20)  # Donchian, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Donchian opposite breakout ---
        if in_position:
            if position_side > 0:  # Long
                # Exit on Donchian(20) lower break (contrarian exit)
                if price <= low_roll[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit on Donchian(20) upper break (contrarian exit)
                if price >= high_roll[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Weekly pivot bias: price > pivot = long bias, price < pivot = short bias
        bullish_bias = price > pivot_1w_aligned[i]
        bearish_bias = price < pivot_1w_aligned[i]
        
        # Donchian breakout conditions with pivot bias alignment
        breakout_long = (price >= high_roll[i]) and bullish_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and bearish_bias and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals