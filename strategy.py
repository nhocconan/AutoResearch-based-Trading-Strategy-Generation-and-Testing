#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Filtered
Hypothesis: In trending markets (1d ADX > 25), breakouts at Camarilla R1/S1 levels continue with the 1d trend. In ranging markets (1d ADX < 25), fade at R1/S1 with mean reversion. Uses volume confirmation (volume > 1.5x 20-period average) to filter false breakouts. Designed for 20-40 trades/year per symbol to avoid fee drag while capturing both range reversals and trend continuations. Works in both bull and bear markets by adapting to regime.
"""

name = "4h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Filtered"
timeframe = "4h"
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
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
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
    
    # --- 1d Volume Average for confirmation ---
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
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
    r1 = pivot + (range_val * 1.1 / 4.0)
    s1 = pivot - (range_val * 1.1 / 4.0)
    r4 = pivot + (range_val * 1.1)
    s4 = pivot - (range_val * 1.1)
    
    # Align Camarilla levels to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # --- 4h Volume Average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for ADX and volume averages
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r4_4h[i]) or np.isnan(s4_4h[i])):
            if position != 0:
                # Check stoploss (2.0x ATR from entry)
                atr_est = np.abs(high_4h[i] - low_4h[i])  # rough 4h ATR estimate
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_1d_aligned[i] < 25
        is_trend = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        if position == 0:
            # Look for entries based on regime
            if is_range and vol_confirm:
                # Mean reversion: fade at R1/S1
                if i > 0:
                    # Rejection at R1 (failed breakout above)
                    if close_4h[i-1] > r1_4h[i-1] and close_4h[i] < r1_4h[i]:
                        signals[i] = -0.25  # short rejection
                        position = -1
                        entry_price = close_4h[i]
                    # Rejection at S1 (failed breakdown below)
                    elif close_4h[i-1] < s1_4h[i-1] and close_4h[i] > s1_4h[i]:
                        signals[i] = 0.25   # long rejection
                        position = 1
                        entry_price = close_4h[i]
            elif is_trend and vol_confirm:
                # Trend following: breakout at R1/S1 continues
                if close_4h[i] > r1_4h[i]:
                    signals[i] = 0.25  # long breakout
                    position = 1
                    entry_price = close_4h[i]
                elif close_4h[i] < s1_4h[i]:
                    signals[i] = -0.25  # short breakdown
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if is_range:
                    # In range, take profit at pivot or S1
                    if close_4h[i] <= pivot[i] or close_4h[i] <= s1_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_4h[i] < s1_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # is_trend
                    # In trend, trail with 1d EMA20 or stop at S4
                    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
                    if not np.isnan(ema20_1d_aligned[i]) and close_4h[i] < ema20_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S4
                    elif close_4h[i] < s4_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if is_range:
                    # In range, take profit at pivot or R1
                    if close_4h[i] >= pivot[i] or close_4h[i] >= r1_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_4h[i] > r1_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # is_trend
                    # In trend, trail with 1d EMA20 or stop at R4
                    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
                    if not np.isnan(ema20_1d_aligned[i]) and close_4h[i] > ema20_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R4
                    elif close_4h[i] > s4_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals