#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Reversal_1dTrend_VolumeFilter
Hypothesis: On 12h timeframe, use daily Camarilla R3/S3 levels for mean reversion in ranging markets (ADX<25) and breakout continuation in trending markets (ADX>25), with volume confirmation to reduce false signals. Designed for 12-37 trades/year to avoid fee drag while capturing both range reversals and trend continuations in BTC/ETH.
"""

name = "12h_Camarilla_R3S3_Reversal_1dTrend_VolumeFilter"
timeframe = "12h"
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
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
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
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1d Volume Moving Average for confirmation ---
    volume_ma_1d = pd.Series(volume_1d := df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
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
    
    # Align Camarilla levels to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for ADX and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or np.isnan(volume_ma_12h[i]) or
            np.isnan(volume_12h[i])):
            if position != 0:
                # Check stoploss (2.0x ATR from entry)
                atr_est = np.abs(high_12h[i] - low_12h[i])  # rough 12h ATR estimate
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x daily average volume
        volume_confirm = volume_12h[i] > 1.5 * volume_ma_12h[i]
        
        # Determine regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_12h[i] < 25
        is_trend = adx_12h[i] > 25
        
        if position == 0:
            # Look for entries based on regime with volume confirmation
            if volume_confirm:
                if is_range:
                    # Mean reversion: fade at R3/S3
                    if i > 0:
                        # Rejection at R3 (failed breakout above)
                        if close_12h[i-1] > r3_12h[i-1] and close_12h[i] < r3_12h[i]:
                            signals[i] = -0.25  # short rejection
                            position = -1
                            entry_price = close_12h[i]
                        # Rejection at S3 (failed breakdown below)
                        elif close_12h[i-1] < s3_12h[i-1] and close_12h[i] > s3_12h[i]:
                            signals[i] = 0.25   # long rejection
                            position = 1
                            entry_price = close_12h[i]
                else:  # is_trend
                    # Trend following: breakout at R4/S4 continues
                    if close_12h[i] > r4_12h[i]:
                        signals[i] = 0.25  # long breakout
                        position = 1
                        entry_price = close_12h[i]
                    elif close_12h[i] < s4_12h[i]:
                        signals[i] = -0.25  # short breakdown
                        position = -1
                        entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if is_range:
                    # In range, take profit at pivot or S3
                    if close_12h[i] <= pivot_12h[i] or close_12h[i] <= s3_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S3
                    elif close_12h[i] < s3_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # is_trend
                    # In trend, trail with 1d EMA20 or stop at S4
                    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema20_12h = align_htf_to_ltf(prices, df_1d, ema20_1d)
                    if not np.isnan(ema20_12h[i]) and close_12h[i] < ema20_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S4
                    elif close_12h[i] < s4_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if is_range:
                    # In range, take profit at pivot or R3
                    if close_12h[i] >= pivot_12h[i] or close_12h[i] >= r3_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R3
                    elif close_12h[i] > r3_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # is_trend
                    # In trend, trail with 1d EMA20 or stop at R4
                    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
                    ema20_12h = align_htf_to_ltf(prices, df_1d, ema20_1d)
                    if not np.isnan(ema20_12h[i]) and close_12h[i] > ema20_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R4
                    elif close_12h[i] > s4_12h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals