#!/usr/bin/env python3
"""
1d Pivot Point (R1/S1) Breakout + Volume Spike + Weekly ADX Trend Filter
Long: Price breaks above R1 (prior week) + volume > 2x 20-day avg + weekly ADX > 25
Short: Price breaks below S1 (prior week) + volume > 2x 20-day avg + weekly ADX > 25
Exit: Price re-enters pivot range (between S1 and R1) or volume drops below average
Uses weekly pivot levels for structure, volume for confirmation, and ADX for trend strength.
Designed to capture breakouts in trending markets while avoiding chop.
Target: 20-60 total trades over 4 years (5-15/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P, R1, R2, S1, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and ADX
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    _, r1_weekly, _, s1_weekly, _ = calculate_pivot_points(
        high_weekly[:-1], low_weekly[:-1], close_weekly[:-1]
    )
    # Shift to align with current week (we use prior week's levels)
    r1_weekly = np.concatenate([ [r1_weekly[0]], r1_weekly[:-1] ])
    s1_weekly = np.concatenate([ [s1_weekly[0]], s1_weekly[:-1] ])
    
    # Align weekly pivot levels to daily timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Calculate weekly ADX(14) for trend strength
    # True Range
    tr1 = np.abs(high_weekly[1:] - low_weekly[1:])
    tr2 = np.abs(high_weekly[1:] - close_weekly[:-1])
    tr3 = np.abs(low_weekly[1:] - close_weekly[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([ [np.nan], tr ])  # align with weekly index
    
    # Directional Movement
    dm_plus = np.where((high_weekly[1:] - high_weekly[:-1]) > (low_weekly[:-1] - low_weekly[1:]), 
                       np.maximum(high_weekly[1:] - high_weekly[:-1], 0), 0)
    dm_minus = np.where((low_weekly[:-1] - low_weekly[1:]) > (high_weekly[1:] - high_weekly[:-1]), 
                        np.maximum(low_weekly[:-1] - low_weekly[1:], 0), 0)
    dm_plus = np.concatenate([ [np.nan], dm_plus ])
    dm_minus = np.concatenate([ [np.nan], dm_minus ])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/14)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr > 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smooth(dx, 14)
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Daily volume average (20-day)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = r1_weekly_aligned[i]
        s1 = s1_weekly_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume spike + strong trend (ADX > 25)
            if price > r1 and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume spike + strong trend (ADX > 25)
            elif price < s1 and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price re-enters pivot range OR weak trend (ADX < 20)
            if price < r1 and price > s1:  # back inside pivot range
                signals[i] = 0.0
                position = 0
            elif adx_val < 20:  # trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price re-enters pivot range OR weak trend (ADX < 20)
            if price < r1 and price > s1:  # back inside pivot range
                signals[i] = 0.0
                position = 0
            elif adx_val < 20:  # trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_PivotPoint_R1S1_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0