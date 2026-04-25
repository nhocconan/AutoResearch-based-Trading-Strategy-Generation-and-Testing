#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDir_VolumeConfirm
Hypothesis: 6h timeframe with Donchian(20) breakouts filtered by weekly pivot direction (from 1w HTF) and volume spikes (>2.0x 20-bar average).
Weekly pivot provides structural bias: long when price above weekly pivot, short when below.
Volume confirmation avoids breakout failures. Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
Works in bull markets via breakout continuation with weekly bias and in bear markets via failed breakout reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for weekly pivot (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot calculation: (weekly high + weekly low + weekly close) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot = typical_price.values
    
    # Align weekly pivot to 6h timeframe (completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss calculation
    tr0 = np.abs(high - low)
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr0[0]], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need enough for Donchian (20), volume MA (20), ATR (14)
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + weekly pivot alignment
            long_breakout = curr_high > donchian_high[i]
            short_breakout = curr_low < donchian_low[i]
            
            # Weekly pivot filter: price must be on correct side of weekly pivot
            long_pivot = curr_close > weekly_pivot_aligned[i]
            short_pivot = curr_close < weekly_pivot_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_pivot)
            short_entry = (short_breakout and volume_spike[i] and short_pivot)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: minimum holding period + exit conditions
            if bars_since_entry < 2:  # Minimum 2 bars (12h) holding period
                signals[i] = 0.25
            else:
                # Exit when price closes below Donchian high (failed breakout) 
                # or weekly pivot breaks or ATR stoploss hit
                atr_stop = entry_price - 3.0 * atr[i]
                if curr_close < donchian_high[i] or curr_close < weekly_pivot_aligned[i] or curr_close < atr_stop:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position: minimum holding period + exit conditions
            if bars_since_entry < 2:  # Minimum 2 bars (12h) holding period
                signals[i] = -0.25
            else:
                # Exit when price closes above Donchian low (failed breakout) 
                # or weekly pivot breaks or ATR stoploss hit
                atr_stop = entry_price + 3.0 * atr[i]
                if curr_close > donchian_low[i] or curr_close > weekly_pivot_aligned[i] or curr_close > atr_stop:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDir_VolumeConfirm"
timeframe = "6h"
leverage = 1.0