#!/usr/bin/env python3
"""
Experiment #4791: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) with volume confirmation (>1.5x average) capture strong momentum moves. Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts above pivot) and bear markets (breakdowns below pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4791_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    # Calculate weekly OHLC from daily data
    if len(df_1d) >= 5:
        # Resample daily to weekly using actual week boundaries
        df_1d_indexed = df_1d.copy()
        df_1d_indexed.index = pd.date_range(
            start=df_1d['open_time'].iloc[0], 
            periods=len(df_1d), 
            freq='1d'
        )
        weekly = df_1d_indexed.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        if len(weekly) > 0:
            # Calculate pivot from prior week (shifted by 1 to avoid look-ahead)
            weekly_high = weekly['high'].shift(1).values
            weekly_low = weekly['low'].shift(1).values
            weekly_close = weekly['close'].shift(1).values
            
            # Weekly pivot formula: (H + L + C) / 3
            weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
            
            # Align weekly pivot to 6h timeframe
            # Need to map each 6h bar to the appropriate weekly pivot value
            weekly_pivot_aligned = np.full(n, np.nan)
            
            # Create weekly timestamp index for alignment
            weekly_timestamps = weekly.index
            if len(weekly_timestamps) > 1:
                # For each 6h bar, find which weekly pivot applies
                price_timestamps = pd.date_range(
                    start=prices['open_time'].iloc[0], 
                    periods=n, 
                    freq='6h'
                )
                
                for i in range(n):
                    current_time = price_timestamps[i]
                    # Find the most recent weekly pivot (prior week's pivot)
                    pivot_idx = np.searchsorted(weekly_timestamps, current_time) - 1
                    if pivot_idx >= 0 and pivot_idx < len(weekly_pivot):
                        weekly_pivot_aligned[i] = weekly_pivot[pivot_idx]
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
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
            
            reverse_long = (price >= high_roll[i]) and (price > weekly_pivot_aligned[i]) and vol_confirm
            reverse_short = (price <= low_roll[i]) and (price < weekly_pivot_aligned[i]) and vol_confirm
            
            if (position_side > 0 and reverse_short) or (position_side < 0 and reverse_long):
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE * position_side
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
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals