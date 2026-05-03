#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Long when price breaks above 6h Donchian upper channel with volume > 2.0x 20-bar average AND weekly pivot > previous week's pivot (bullish bias)
# Short when price breaks below 6h Donchian lower channel with volume > 2.0x 20-bar average AND weekly pivot < previous week's pivot (bearish bias)
# Exit via ATR trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR, short exit when price > lowest_low_since_entry + 2.0 * ATR
# Donchian provides structure, weekly pivot gives higher-timeframe bias, volume confirms conviction.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee drag.

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot direction
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot (typical price) and its direction
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    # Pivot direction: 1 if current pivot > previous pivot, -1 if <, 0 if equal
    pivot_direction = np.where(typical_price_1w > np.roll(typical_price_1w, 1), 1,
                              np.where(typical_price_1w < np.roll(typical_price_1w, 1), -1, 0))
    # Set first value to 0 (no previous week)
    pivot_direction[0] = 0
    pivot_direction_aligned = align_htf_to_ltf(prices, df_1w, pivot_direction)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(20, 20) + 1  # Donchian(20) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(pivot_direction_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper channel with volume spike AND bullish weekly pivot
            if (close[i] > highest_20[i] and 
                volume_spike[i] and pivot_direction_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below Donchian lower channel with volume spike AND bearish weekly pivot
            elif (close[i] < lowest_20[i] and 
                  volume_spike[i] and pivot_direction_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals