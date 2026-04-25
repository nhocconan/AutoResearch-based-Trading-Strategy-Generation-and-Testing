#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
Hypothesis: On 6h timeframe, Donchian channel breakouts capture momentum moves.
Filtering by 1d weekly pivot direction (price above/below weekly pivot) ensures
trades align with higher-timeframe bias. Volume confirmation avoids low-conviction
breakouts. Works in bull/bear via pivot-based trend filter. Targets 12-37 trades/year.
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
    
    # Get 1d data for weekly pivot calculation (using weekly OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly OHLC from daily data
    # Group into weeks (5 trading days approx) - use rolling window for simplicity
    # Weekly high = max of last 5 daily highs, weekly low = min of last 5 daily lows
    # Weekly close = last daily close, weekly open = first daily open of the week
    # For pivot calculation, we need typical price: (H+L+C)/3
    
    # Calculate rolling weekly OHLC (5-day window)
    weekly_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close).rolling(window=5, min_periods=5).apply(lambda x: x[-1], raw=True).values
    weekly_open = pd.Series(close).rolling(window=5, min_periods=5).apply(lambda x: x[0], raw=True).values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (using previous week's value to avoid look-ahead)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Get 6h data for Donchian channel (20-period)
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR for stop loss (14 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian (20) and ATR (14)
    start_idx = max(lookback, 14, 20)  # 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_value = atr[i]
        vol_ma = vol_ma_20[i]
        pivot = weekly_pivot_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        # Volume spike condition
        volume_spike = curr_volume > 1.5 * vol_ma
        
        # Pivot direction: price above/below weekly pivot
        above_pivot = curr_close > pivot
        below_pivot = curr_close < pivot
        
        # Donchian breakout conditions
        bullish_breakout = curr_close > upper
        bearish_breakout = curr_close < lower
        
        # Update tracking variables for trailing stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or pivot reversal
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.0*ATR from highest since entry
                if curr_close < highest_since_entry - 2.0 * atr_value:
                    exit_signal = True
                # Pivot reversal (price crosses below weekly pivot)
                elif curr_close < pivot:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.0 * atr_value:
                    exit_signal = True
                # Pivot reversal (price crosses above weekly pivot)
                elif curr_close > pivot:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Donchian breakout + pivot alignment + volume
        if position == 0:
            # Long: Donchian breakout above upper AND price above weekly pivot
            long_condition = bullish_breakout and above_pivot and volume_spike
            # Short: Donchian breakout below lower AND price below weekly pivot
            short_condition = bearish_breakout and below_pivot and volume_spike
            
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