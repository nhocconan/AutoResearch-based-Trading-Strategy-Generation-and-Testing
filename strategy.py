#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for weekly pivot calculation (R4/S4 levels) and volume average.
- Donchian Channel: identifies breakouts from 20-period price channels.
- Entry: Long when price breaks above Donchian upper AND price > 1d weekly R4 pivot AND volume > 1.5 * 20-period average volume.
         Short when price breaks below Donchian lower AND price < 1d weekly S4 pivot AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout signal or price retouches the Donchian midpoint (mean reversion within channel).
- Signal size: 0.25 discrete to minimize fee drag.
- Weekly pivots from 1d provide robust structural levels that work in both bull and bear markets.
- Volume confirmation ensures breakout legitimacy and filters false breakouts.
- Donchian breakouts capture momentum while pivot levels provide institutional reference points.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period=20):
    """Calculate Donchian Channels: returns upper, lower, middle."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def calculate_weekly_pivots(df_1d):
    """
    Calculate weekly pivot points from daily data.
    Uses the prior week's high, low, close to calculate pivot levels.
    Returns R4, R3, R2, R1, PP, S1, S2, S3, S4 levels.
    """
    if len(df_1d) < 5:
        return {
            'R4': np.full(len(df_1d), np.nan),
            'R3': np.full(len(df_1d), np.nan),
            'R2': np.full(len(df_1d), np.nan),
            'R1': np.full(len(df_1d), np.nan),
            'PP': np.full(len(df_1d), np.nan),
            'S1': np.full(len(df_1d), np.nan),
            'S2': np.full(len(df_1d), np.nan),
            'S3': np.full(len(df_1d), np.nan),
            'S4': np.full(len(df_1d), np.nan)
        }
    
    # Resample daily data to weekly (using Friday as week end)
    # We'll calculate pivots based on prior week's HLC
    high_weekly = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week's high
    low_weekly = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)   # Prior week's low
    close_weekly = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)  # Prior week's close
    
    # Calculate pivot point
    pp = (high_weekly + low_weekly + close_weekly) / 3.0
    
    # Calculate support and resistance levels
    r1 = 2 * pp - low_weekly
    s1 = 2 * pp - high_weekly
    r2 = pp + (high_weekly - low_weekly)
    s2 = pp - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pp - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pp)
    r4 = pp + 3 * (high_weekly - low_weekly)
    s4 = pp - 3 * (high_weekly - low_weekly)
    
    return {
        'R4': r4.values,
        'R3': r3.values,
        'R2': r2.values,
        'R1': r1.values,
        'PP': pp.values,
        'S1': s1.values,
        'S2': s2.values,
        'S3': s3.values,
        'S4': s4.values
    }

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d weekly pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need sufficient data for weekly pivots
        return np.zeros(n)
    
    weekly_pivots = calculate_weekly_pivots(df_1d)
    r4_1d = weekly_pivots['R4']
    s4_1d = weekly_pivots['S4']
    
    # Align weekly pivot levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Donchian Channels from 6h data (20-period)
    dc_upper, dc_lower, dc_middle = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need 20 for Donchian, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(dc_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below Donchian lower OR price retouches Donchian middle (mean reversion)
            if position == 1:
                if curr_close < dc_lower[i] or abs(curr_close - dc_middle[i]) < (dc_upper[i] - dc_lower[i]) * 0.05:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper OR price retouches Donchian middle (mean reversion)
            elif position == -1:
                if curr_close > dc_upper[i] or abs(curr_close - dc_middle[i]) < (dc_upper[i] - dc_lower[i]) * 0.05:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with pivot filter and volume confirmation
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_close > dc_upper[i] and prev_close <= dc_upper[i-1]
            breakout_down = curr_close < dc_lower[i] and prev_close >= dc_lower[i-1]
            
            # Pivot filter: price vs weekly R4/S4 levels
            long_pivot = curr_close > r4_1d_aligned[i]
            short_pivot = curr_close < s4_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if breakout_up and long_pivot and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif breakout_down and short_pivot and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dWeeklyPivot_R4S4_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0