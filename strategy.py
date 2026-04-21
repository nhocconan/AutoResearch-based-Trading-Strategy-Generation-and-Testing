#!/usr/bin/env python3
"""
12h strategy: Weekly 3-period ATR trailing stop with Monday 00:00 UTC open price filter.
Buy when price > Monday open + 0.5*weekly ATR, sell when price < Monday open - 0.5*weekly ATR.
Trailing stop: exit long at 3*ATR below highest close since entry, exit short at 3*ATR above lowest close since entry.
Designed for ~15-25 trades/year to minimize fee drag while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Calculate weekly ATR (3-period)
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_w = pd.Series(tr).rolling(window=3, min_periods=3).mean().values
    
    # Align weekly ATR to 12h
    atr_w_aligned = align_htf_to_ltf(prices, df_1w, atr_w)
    
    # Get Monday 00:00 UTC open price for each week
    # Create boolean mask for Monday 00:00 UTC in weekly data
    open_time_w = df_1w['open_time'].values
    # Convert to pandas Timestamp to get weekday
    monday_mask = pd.Series(open_time_w).dt.weekday == 0  # Monday=0
    monday_mask &= pd.Series(open_time_w).dt.hour == 0
    monday_mask &= pd.Series(open_time_w).dt.minute == 0
    
    # Get Monday open prices
    monday_open = np.where(monday_mask, df_1w['open'].values, np.nan)
    # Forward fill to get the Monday open for the entire week
    monday_open_ff = pd.Series(monday_open).ffill().values
    # Align to 12h
    monday_open_aligned = align_htf_to_ltf(prices, df_1w, monday_open_ff)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_close_since_entry = np.full(n, np.nan)
    lowest_close_since_entry = np.full(n, np.nan)
    
    for i in range(10, n):
        # Skip if indicators not ready
        if np.isnan(atr_w_aligned[i]) or np.isnan(monday_open_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_close_since_entry[i] = np.nan
                lowest_close_since_entry[i] = np.nan
            continue
        
        price_close = prices['close'].iloc[i]
        atr_val = atr_w_aligned[i]
        monday_open_val = monday_open_aligned[i]
        
        if position == 0:
            # Enter long: price > Monday open + 0.5*ATR
            if price_close > monday_open_val + 0.5 * atr_val:
                signals[i] = 0.25
                position = 1
                highest_close_since_entry[i] = price_close
            # Enter short: price < Monday open - 0.5*ATR
            elif price_close < monday_open_val - 0.5 * atr_val:
                signals[i] = -0.25
                position = -1
                lowest_close_since_entry[i] = price_close
        
        elif position == 1:
            # Update highest close since entry
            highest_close_since_entry[i] = max(highest_close_since_entry[i-1], price_close)
            # Exit conditions:
            # 1. Trailing stop: 3*ATR below highest close
            # 2. Price < Monday open - 0.5*ATR (reverse signal)
            trail_stop = highest_close_since_entry[i] - 3.0 * atr_val
            reverse_signal = price_close < monday_open_val - 0.5 * atr_val
            
            if price_close < trail_stop or reverse_signal:
                signals[i] = 0.0
                position = 0
                highest_close_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest close since entry
            lowest_close_since_entry[i] = min(lowest_close_since_entry[i-1], price_close)
            # Exit conditions:
            # 1. Trailing stop: 3*ATR above lowest close
            # 2. Price > Monday open + 0.5*ATR (reverse signal)
            trail_stop = lowest_close_since_entry[i] + 3.0 * atr_val
            reverse_signal = price_close > monday_open_val + 0.5 * atr_val
            
            if price_close > trail_stop or reverse_signal:
                signals[i] = 0.0
                position = 0
                lowest_close_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_W_ATR_Trail_Stop_MondayOpenFilter"
timeframe = "12h"
leverage = 1.0