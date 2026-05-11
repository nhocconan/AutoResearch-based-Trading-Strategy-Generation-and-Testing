#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_R3S3_RangeReversion_1dTrend
Hypothesis: In ranging markets (ADX<25), price reverts to the mean from Camarilla R3/S3 levels. In trending markets (ADX>25), breakouts at R4/S4 continue with the 1d trend. Uses ADX regime filter to switch between mean reversion and trend following. Designed for 15-35 trades/year per symbol to avoid fee drag while capturing both range reversals and trend continuations.
"""

name = "6h_Camarilla_Pivot_R3S3_RangeReversion_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # --- 1d ADX for regime detection (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1d Camarilla Pivot Levels (using previous day) ---
    # Calculate from previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 2.0)
    s3 = pivot - (range_val * 1.1 / 2.0)
    r4 = pivot + (range_val * 1.1)
    s4 = pivot - (range_val * 1.1)
    
    # Align Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # --- 6h Close for price reference ---
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i])):
            if position != 0:
                # Check stoploss (1.5x ATR from entry)
                atr_est = np.abs(high_6h[i] - low_6h[i])  # rough 6m ATR estimate
                if position == 1 and close_6h[i] <= entry_price - 1.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 1.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_1d_aligned[i] < 25
        is_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Look for entries based on regime
            if is_range:
                # Mean reversion: fade at R3/S3
                if close_6h[i] <= r3_6h[i] and close_6h[i] > s3_6h[i]:
                    # In range, look for rejection at levels
                    if i > 0:
                        # Rejection at R3 (failed breakout above)
                        if close_6h[i-1] > r3_6h[i-1] and close_6h[i] < r3_6h[i]:
                            signals[i] = -0.25  # short rejection
                            position = -1
                            entry_price = close_6h[i]
                        # Rejection at S3 (failed breakdown below)
                        elif close_6h[i-1] < s3_6h[i-1] and close_6h[i] > s3_6h[i]:
                            signals[i] = 0.25   # long rejection
                            position = 1
                            entry_price = close_6h[i]
            else:  # is_trend
                # Trend following: breakout at R4/S4 continues
                if close_6h[i] > r4_6h[i]:
                    signals[i] = 0.25  # long breakout
                    position = 1
                    entry_price = close_6h[i]
                elif close_6h[i] < s4_6h[i]:
                    signals[i] = -0.25  # short breakdown
                    position = -1
                    entry_price = close_6h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if is_range:
                    # In range, take profit at pivot or S3
                    if close_6h[i] <= pivot[i] or close_6h[i] <= s3_6h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S3
                    elif close_6h[i] < s3_6h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # is_trend
                    # In trend, trail with 1d EMA20 or stop at S4
                    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
                    if not np.isnan(ema20_1d_aligned[i]) and close_6h[i] < ema20_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S4
                    elif close_6h[i] < s4_6h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if is_range:
                    # In range, take profit at pivot or R3
                    if close_6h[i] >= pivot[i] or close_6h[i] >= r3_6h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R3
                    elif close_6h[i] > r3_6h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # is_trend
                    # In trend, trail with 1d EMA20 or stop at R4
                    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
                    if not np.isnan(ema20_1d_aligned[i]) and close_6h[i] > ema20_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R4
                    elif close_6h[i] > r4_6h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals