#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
Hypothesis: Daily weekly pivot (based on prior week's H/L/C) determines institutional bias.
6h Donchian breakout in direction of weekly pivot with volume confirmation captures
trend continuation while avoiding counter-trend whipsaws. Works in bull/bear by
adapting pivot bias: long when price above weekly pivot, short when below.
Uses discrete sizing (0.25) and volume threshold (1.8x) to target ~80-120 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's OHLC (using 5-day approximation)
    # Weekly high = max of prior 5 daily highs
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    # Weekly low = min of prior 5 daily lows
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    # Weekly close = prior Friday's close (approximated as 5th prior daily close)
    weekly_close = pd.Series(df_1d['close']).shift(5).values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align daily data to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 6h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for dynamic stop (optional trailing stop via signal=0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for weekly pivot (5d) + Donchian (20) + VolMA (20) + ATR (14)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        pivot_level = weekly_pivot_6h[i]
        atr_value = atr[i]
        
        # Volume spike: current volume > 1.8 * 20-period average
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        bullish_breakout = curr_close > donchian_high[i]  # Break above upper band
        bearish_breakout = curr_close < donchian_low[i]   # Break below lower band
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.5*ATR from highest since entry
                if curr_close < highest_since_entry - 2.5 * atr_value:
                    exit_signal = True
                # Reverse breakout or pivot rejection
                elif curr_close < donchian_low[i] or curr_close < pivot_level:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.5*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.5 * atr_value:
                    exit_signal = True
                # Reverse breakout or pivot rejection
                elif curr_close > donchian_high[i] or curr_close > pivot_level:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Donchian breakout + pivot alignment + volume
        if position == 0:
            # Long: break above Donchian high AND price above weekly pivot
            long_condition = bullish_breakout and (curr_close > pivot_level) and volume_spike
            # Short: break below Donchian low AND price below weekly pivot
            short_condition = bearish_breakout and (curr_close < pivot_level) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dWeeklyPivot_Direction_Volume_v1"
timeframe = "6h"
leverage = 1.0