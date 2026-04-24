#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and weekly pivot bias.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter (price above/below daily EMA) and 1w for weekly pivot bias.
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND close > weekly pivot;
         Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND close < weekly pivot.
- Exit: ATR(14) trailing stop (long: highest_high - 2.0*ATR; short: lowest_low + 2.0*ATR).
- Signal size: 0.25 discrete to minimize fee drag while capturing trends.
- Designed to work in both bull and bear markets by aligning with higher timeframe trend and structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Donchian(20) on 6h using previous period's high/low
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(14) on 6h for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_6h[0] - low_6h[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align all indicators to primary 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track extreme prices for trailing stop
    highest_high = 0.0
    lowest_low = float('inf')
    
    # Start from index where all indicators are ready (max of 20 for Donchian, 14 for ATR, 50 for EMA)
    start_idx = max(20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = float('inf')
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Update trailing stop extremes
        if position == 1:  # long
            if curr_high > highest_high:
                highest_high = curr_high
        elif position == -1:  # short
            if curr_low < lowest_low:
                lowest_low = curr_low
        
        # Calculate stop levels
        long_stop = highest_high - 2.0 * atr_aligned[i] if highest_high > 0 else 0.0
        short_stop = lowest_low + 2.0 * atr_aligned[i] if lowest_low != float('inf') else float('inf')
        
        # Check for stoploss
        if position == 1 and curr_close < long_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        elif position == -1 and curr_close > short_stop:
            signals[i] = 0.0
            position = 0
            highest_high = 0.0
            lowest_low = float('inf')
            continue
        
        # Breakout conditions
        bullish_breakout = curr_close > donchian_high_aligned[i]
        bearish_breakout = curr_close < donchian_low_aligned[i]
        
        # Trend filter from 1d EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Weekly pivot bias
        close_above_pivot = curr_close > weekly_pivot_aligned[i]
        close_below_pivot = curr_close < weekly_pivot_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if bullish_breakout and price_above_ema and close_above_pivot:
                signals[i] = 0.25
                position = 1
                highest_high = curr_high
                lowest_low = float('inf')
            elif bearish_breakout and price_below_ema and close_below_pivot:
                signals[i] = -0.25
                position = -1
                highest_high = 0.0
                lowest_low = curr_low
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_1wPivotBias_ATRStop_v1"
timeframe = "6h"
leverage = 1.0