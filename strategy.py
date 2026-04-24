#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
- Regime: ADX > 25 = trending (follow Elder Ray signals), ADX < 20 = range (fade Elder Ray extremes).
- Entry: Long when Bull Power > 0 and Bull Power rising (bullish momentum) in trending regime.
         Short when Bear Power > 0 and Bear Power rising (bearish momentum) in trending regime.
         In range regime: fade extreme Bull/Bear Power (mean reversion at 2 std dev).
- Exit: Opposite Elder Ray signal or return to EMA13.
- Works in bull via buying strength in uptrend, in bear via selling weakness in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_elder_ray(high, low, close, ema13):
    """Calculate Elder Ray Bull Power and Bear Power"""
    bull_power = high - ema13
    bear_power = ema13 - low
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX and EMA13
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime filter (ADX > 25 = trending, < 20 = range)
    # ADX calculation: +DI, -DI, DX, then ADX smoothed
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Rest: Wilder smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # +DI and -DI
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 14:
        # First ADX: average of first 14 DX values
        adx[13] = np.nanmean(dx[1:15])  # Skip first NaN DX
        # Rest: Wilder smoothing of DX
        alpha = 1.0 / 14
        for i in range(15, len(dx)):
            adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Calculate Elder Ray for each 1d bar
    bull_power_1d = np.full(len(df_1d), np.nan)
    bear_power_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        bull_power, bear_power = calculate_elder_ray(
            df_1d['high'].iloc[i],
            df_1d['low'].iloc[i],
            df_1d['close'].iloc[i],
            ema13_1d[i]
        )
        bull_power_1d[i] = bull_power
        bear_power_1d[i] = bear_power
    
    # Align 1d indicators to 6h
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Calculate 6h EMA13 for exit reference
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 13)  # Need enough 1d bars for ADX/EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema13_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals based on regime
            if volume_spike[i]:
                if adx_aligned[i] > 25:  # Trending regime - follow momentum
                    # Bullish: Bull Power > 0 and rising (current > previous)
                    if bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish: Bear Power > 0 and rising (current > previous)
                    elif bear_power_aligned[i] > 0 and bear_power_aligned[i] > bear_power_aligned[i-1]:
                        signals[i] = -0.25
                        position = -1
                elif adx_aligned[i] < 20:  # Range regime - fade extremes
                    # Calculate 20-period std dev of Elder Ray for extreme detection
                    # Use aligned 6h Elder Ray approximated from 1d (simplified)
                    # Fade when Bull/Bear Power > 2 std dev above/below mean
                    # For simplicity, use fixed thresholds based on typical 1d ranges
                    bull_mean = np.nanmean(bull_power_aligned[max(0, i-100):i]) if i >= 100 else 0
                    bear_mean = np.nanmean(bear_power_aligned[max(0, i-100):i]) if i >= 100 else 0
                    bull_std = np.nanstd(bull_power_aligned[max(0, i-100):i]) if i >= 100 else 1
                    bear_std = np.nanstd(bear_power_aligned[max(0, i-100):i]) if i >= 100 else 1
                    
                    # Fade extreme Bull Power (overbought)
                    if bull_power_aligned[i] > bull_mean + 2 * bull_std:
                        signals[i] = -0.25  # Short mean reversion
                        position = -1
                    # Fade extreme Bear Power (oversold)
                    elif bear_power_aligned[i] > bear_mean + 2 * bear_std:
                        signals[i] = 0.25   # Long mean reversion
                        position = 1
        elif position == 1:
            # Long exit: Bear Power > 0 (momentum shift) or price returns to EMA13
            if bear_power_aligned[i] > 0 or close[i] <= ema13_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power > 0 (momentum shift) or price returns to EMA13
            if bull_power_aligned[i] > 0 or close[i] >= ema13_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0