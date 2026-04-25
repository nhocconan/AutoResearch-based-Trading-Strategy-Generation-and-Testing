#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyRegime_v1
Hypothesis: Trade 6h Camarilla R3/S3 breakouts with 1d EMA34 trend filter and weekly regime filter based on ADX.
In trending markets (weekly ADX > 25): trade breakout continuation at R4/S4 levels.
In ranging markets (weekly ADX <= 25): trade mean reversion at R3/S3 levels.
Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get weekly data for regime filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # We need to shift by 1 to avoid look-ahead
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    r4 = pivot + (range_ * 1.1 / 2)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate 6h Donchian(5) for breakout confirmation (shorter for quicker signals)
    high_ma_5 = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_ma_5 = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and ADX (14+14=28) and Donchian (5)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(high_ma_5[i]) or
            np.isnan(low_ma_5[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above 1d EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Determine weekly regime: trending if ADX > 25, ranging if ADX <= 25
        weekly_trending = adx_aligned[i] > 25
        weekly_ranging = adx_aligned[i] <= 25
        
        if position == 0:
            if weekly_trending:
                # Trending market: trade breakout continuation at R4/S4
                long_setup = (close[i] > r4_aligned[i]) and htf_1d_bullish and (close[i] > high_ma_5[i-1])
                short_setup = (close[i] < s4_aligned[i]) and htf_1d_bearish and (close[i] < low_ma_5[i-1])
            else:
                # Ranging market: trade mean reversion at R3/S3
                long_setup = (close[i] < s3_aligned[i]) and (close[i] > low_ma_5[i-1])  # Oversold bounce
                short_setup = (close[i] > r3_aligned[i]) and (close[i] < high_ma_5[i-1])  # Overbought rejection
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if weekly_trending:
                # In trending market: exit on trend reversal or touch of opposite S4
                exit_signal = (not htf_1d_bullish) or (close[i] < s4_aligned[i])
            else:
                # In ranging market: exit on mean reversion to pivot or touch of R3
                exit_signal = (close[i] > pivot_aligned[i]) or (close[i] > r3_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if weekly_trending:
                # In trending market: exit on trend reversal or touch of opposite R4
                exit_signal = htf_1d_bullish or (close[i] > r4_aligned[i])
            else:
                # In ranging market: exit on mean reversion to pivot or touch of S3
                exit_signal = (close[i] < pivot_aligned[i]) or (close[i] < s3_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyRegime_v1"
timeframe = "6h"
leverage = 1.0