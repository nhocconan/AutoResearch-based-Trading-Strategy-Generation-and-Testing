#!/usr/bin/env python3
"""
6h_1w_1d_OrderBlock_Flow_v1
Hypothesis: Combine weekly order blocks with daily momentum on 6h timeframe.
Long when price breaks above bullish order block (from weekly swing low) with daily RSI > 50,
short when breaks below bearish order block (from weekly swing high) with daily RSI < 50.
Order blocks represent institutional supply/demand zones; RSI filters for momentum alignment.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
Works in bull via continuation from demand zones, in bear via reversal at supply zones.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_OrderBlock_Flow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for swing points and order blocks
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily data for RSI momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate weekly swing high/low for order blocks (using 5-bar lookback)
    # Swing high: high surrounded by lower highs on both sides
    # Swing low: low surrounded by higher lows on both sides
    swing_high = np.full(len(df_1w), np.nan)
    swing_low = np.full(len(df_1w), np.nan)
    
    for i in range(2, len(df_1w) - 2):
        # Swing high condition
        if (df_1w['high'].iloc[i] > df_1w['high'].iloc[i-1] and 
            df_1w['high'].iloc[i] > df_1w['high'].iloc[i-2] and
            df_1w['high'].iloc[i] > df_1w['high'].iloc[i+1] and
            df_1w['high'].iloc[i] > df_1w['high'].iloc[i+2]):
            swing_high[i] = df_1w['high'].iloc[i]
        
        # Swing low condition
        if (df_1w['low'].iloc[i] < df_1w['low'].iloc[i-1] and 
            df_1w['low'].iloc[i] < df_1w['low'].iloc[i-2] and
            df_1w['low'].iloc[i] < df_1w['low'].iloc[i+1] and
            df_1w['low'].iloc[i] < df_1w['low'].iloc[i+2]):
            swing_low[i] = df_1w['low'].iloc[i]
    
    # Forward fill swing points to create order block zones
    swing_high_series = pd.Series(swing_high)
    swing_low_series = pd.Series(swing_low)
    ob_high = swing_high_series.ffill().bfill().values  # Bearish OB (from swing high)
    ob_low = swing_low_series.ffill().bfill().values    # Bullish OB (from swing low)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly order blocks and daily RSI to 6h timeframe
    ob_high_aligned = align_htf_to_ltf(prices, df_1w, ob_high)
    ob_low_aligned = align_htf_to_ltf(prices, df_1w, ob_low)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(ob_high_aligned[i]) or np.isnan(ob_low_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: price breaks order block with RSI filter
        long_entry = (close[i] > ob_high_aligned[i] and rsi_aligned[i] > 50)
        short_entry = (close[i] < ob_low_aligned[i] and rsi_aligned[i] < 50)
        
        # Exit conditions: price returns to opposite order block
        long_exit = close[i] < ob_low_aligned[i]
        short_exit = close[i] > ob_high_aligned[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals