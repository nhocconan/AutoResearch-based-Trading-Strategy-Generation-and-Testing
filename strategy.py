#!/usr/bin/env python3
"""
Experiment #031: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Combines 6h Donchian(20) breakouts with 1d weekly pivot levels (from prior week) 
to determine institutional bias, confirmed by volume spikes. In bullish weekly bias (price 
above weekly pivot), we take long breakouts above Donchian upper band. In bearish bias 
(price below weekly pivot), we take short breakouts below Donchian lower band. Volume 
confirmation ensures institutional participation. Uses discrete position sizing (0.25) to 
minimize fee drag. Target: 75-150 trades over 4 years. Works in both bull/bear via pivot 
filter that adapts to weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_031_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot from prior week's OHLC
    # Weekly high = max(high) over last 7 days (approximated as last 7 daily bars)
    # Weekly low = min(low) over last 7 days
    # Weekly close = close of 7th bar ago
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=7, min_periods=7).max().values
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=7, min_periods=7).min().values
    weekly_close = pd.Series(df_1d['close'].values).shift(7).values  # Prior week close
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 6h Indicators: Volume Spike (2x 20-period average) ===
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20) + 7  # Donchian(20) + weekly lookback
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = volume_ma[i]
        
        # --- Weekly Bias from Pivot ---
        weekly_bullish = price > weekly_pivot_aligned[i] * 1.001  # Above pivot with small buffer
        weekly_bearish = price < weekly_pivot_aligned[i] * 0.999  # Below pivot with small buffer
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > donchian_high[i]  # New 20-period high
        breakout_down = price < donchian_low[i]  # New 20-period low
        
        # --- Volume Confirmation ---
        vol_confirmed = volume[i] > (vol_ma * 2.0)
        
        # --- Exit Logic: Time-based or opposite signal ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit on bearish weekly flip OR Donchian breakdown OR max 12 bars hold
                if weekly_bearish or breakout_down or bars_since_entry >= 12:
                    exit_signal = True
            else:  # Short position
                # Exit on bullish weekly flip OR Donchian breakout OR max 12 bars hold
                if weekly_bullish or breakout_up or bars_since_entry >= 12:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Weekly bullish bias + Donchian breakout up + volume confirmation
        if weekly_bullish and breakout_up and vol_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Weekly bearish bias + Donchian breakout down + volume confirmation
        elif weekly_bearish and breakout_down and vol_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
    
    return signals