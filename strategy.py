#!/usr/bin/env python3
"""
6h_FibonacciExtension_Retest
Hypothesis: On 6H timeframe, price often respects Fibonacci extension levels (127.2%, 161.8%) of the prior swing. 
In trending markets (ADX > 25 on 1D), retests of these extensions with volume confirmation offer high-probability entries in the trend direction. 
In ranging markets (ADX < 20), fade at extensions for mean reversion. Uses weekly trend filter (price above/below weekly 200 EMA) to avoid counter-trend trades.
Targets 50-150 total trades over 4 years.
"""

name = "6h_FibonacciExtension_Retest"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1D data for ADX and swing calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 6H OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1D ADX for trend strength (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- Swing detection for Fibonacci extensions (using 1D pivots) ---
    # Find swing highs and lows using 3-bar lookback/forward
    swing_high = np.zeros_like(high_1d, dtype=bool)
    swing_low = np.zeros_like(low_1d, dtype=bool)
    
    for i in range(2, len(high_1d)-2):
        if (high_1d[i] >= high_1d[i-1] and high_1d[i] >= high_1d[i-2] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            swing_high[i] = True
        if (low_1d[i] <= low_1d[i-1] and low_1d[i] <= low_1d[i-2] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            swing_low[i] = True
    
    # Get the most recent swing points
    last_swing_high_idx = np.where(swing_high)[0]
    last_swing_low_idx = np.where(swing_low)[0]
    
    # Arrays to store extension levels
    fib_ext_127 = np.full_like(close_1d, np.nan)
    fib_ext_161 = np.full_like(close_1d, np.nan)
    
    # Calculate extensions when we have both swing points
    if len(last_swing_high_idx) > 0 and len(last_swing_low_idx) > 0:
        # Use the most recent complete swing (high then low or low then high)
        # For simplicity, use the closest pair
        for i in range(len(close_1d)):
            # Find most recent swing high and low before current index
            swing_high_before = last_swing_high_idx[last_swing_high_idx < i]
            swing_low_before = last_swing_low_idx[last_swing_low_idx < i]
            
            if len(swing_high_before) > 0 and len(swing_low_before) > 0:
                # Determine swing direction based on order
                last_high = swing_high_before[-1]
                last_low = swing_low_before[-1]
                
                if last_high > last_low:  # Recent high after low = bullish swing
                    swing_low_price = low_1d[last_low]
                    swing_high_price = high_1d[last_high]
                    swing_range = swing_high_price - swing_low_price
                    # Fibonacci extensions above swing high
                    fib_ext_127[i] = swing_high_price + 0.272 * swing_range
                    fib_ext_161[i] = swing_high_price + 0.618 * swing_range
                else:  # Recent low after high = bearish swing
                    swing_high_price = high_1d[last_high]
                    swing_low_price = low_1d[last_low]
                    swing_range = swing_high_price - swing_low_price
                    # Fibonacci extensions below swing low
                    fib_ext_127[i] = swing_low_price - 0.272 * swing_range
                    fib_ext_161[i] = swing_low_price - 0.618 * swing_range
    
    # Align Fibonacci levels to 6H
    fib_ext_127_6h = align_htf_to_ltf(prices, df_1d, fib_ext_127)
    fib_ext_161_6h = align_htf_to_ltf(prices, df_1d, fib_ext_161)
    
    # --- Weekly trend filter (price vs weekly 200 EMA) ---
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_6h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    weekly_uptrend = close > ema200_1w_6h
    weekly_downtrend = close < ema200_1w_6h
    
    # --- 6H Volume confirmation ---
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_6h[i]) or np.isnan(fib_ext_127_6h[i]) or 
            np.isnan(fib_ext_161_6h[i]) or np.isnan(ema200_1w_6h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Market regime
        trending = adx_6h[i] > 25
        ranging = adx_6h[i] < 20
        
        if position == 0:
            # Look for new entries
            if trending and vol_confirm:
                # In trending market, look for retests of Fibonacci extensions
                if weekly_uptrend:
                    # Long on retest of 127.2% or 161.8% extension as support
                    if (abs(close[i] - fib_ext_127_6h[i]) < 0.005 * close[i] or 
                        abs(close[i] - fib_ext_161_6h[i]) < 0.005 * close[i]):
                        signals[i] = 0.25
                        position = 1
                elif weekly_downtrend:
                    # Short on retest of extensions as resistance
                    if (abs(close[i] - fib_ext_127_6h[i]) < 0.005 * close[i] or 
                        abs(close[i] - fib_ext_161_6h[i]) < 0.005 * close[i]):
                        signals[i] = -0.25
                        position = -1
            elif ranging and vol_confirm:
                # In ranging market, fade at extensions (mean reversion)
                if close[i] > fib_ext_161_6h[i]:
                    # Price above upper extension, expect reversion down
                    signals[i] = -0.25
                    position = -1
                elif close[i] < fib_ext_127_6h[i]:
                    # Price below lower extension, expect reversion up
                    signals[i] = 0.25
                    position = 1
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit if price breaks below 127.2% extension or weekly trend fails
                if close[i] < fib_ext_127_6h[i] or not weekly_uptrend[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit if price breaks above 161.8% extension or weekly trend fails
                if close[i] > fib_ext_161_6h[i] or not weekly_downtrend[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals