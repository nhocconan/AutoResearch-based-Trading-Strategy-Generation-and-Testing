#!/usr/bin/env python3
"""
exp_6691_6h_donchian20_1d_pivot_dir_v1
Hypothesis: 6h Donchian(20) breakout with 1-day pivot direction filter and volume confirmation.
Only trade breakouts in the direction of the 1-day trend (above/below pivot). 
In ranging markets (near pivot), fade reversals at Donchian bands with volume exhaustion.
Designed for 6h timeframe to capture swings with ~15-35 trades/year.
Uses discrete position sizing (0.25) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6691_6h_donchian20_1d_pivot_dir_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DC_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
VOL_EXIT_THRESHOLD = 0.6
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day pivot (classic: (H+L+C)/3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 for lookback - use completed day only)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align HTF pivot to LTF (6h) with shift(1) for completed days only
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    dc_high = pd.Series(high).rolling(window=DC_PERIOD, min_periods=DC_PERIOD).max().values
    dc_low = pd.Series(low).rolling(window=DC_PERIOD, min_periods=DC_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DC_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if data not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine market regime based on 1d pivot
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        
        # Volume conditions
        vol_high = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
        vol_low = volume[i] < vol_ma[i] * VOL_EXIT_THRESHOLD
        
        # Breakout signals (only in direction of 1d pivot)
        long_breakout = above_pivot and (close[i] > dc_high[i]) and vol_high
        short_breakout = below_pivot and (close[i] < dc_low[i]) and vol_high
        
        # Mean reversion signals (fade Donchian extremes when near pivot)
        # Long when price touches lower band AND is above pivot (bullish bias)
        long_mean_revert = above_pivot and (close[i] <= dc_low[i]) and vol_low
        # Short when price touches upper band AND is below pivot (bearish bias)
        short_mean_revert = below_pivot and (close[i] >= dc_high[i]) and vol_low
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout or long_mean_revert:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout or short_mean_revert:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals