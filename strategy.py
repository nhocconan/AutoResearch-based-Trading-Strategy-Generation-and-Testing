#!/usr/bin/env python3
"""
6h_WeeklyPivot_Confluence_Breakout_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, weekly Camarilla pivot breakouts (R4/S4) combined with 1d EMA trend filter and volume confirmation.
Weekly pivots capture major support/resistance from prior week, reducing false breakouts. 1d EMA ensures alignment with intermediate trend.
Volume spike confirms institutional participation. Target: 12-30 trades/year (50-150 over 4 years).
Designed to work in both bull (breakout continuation) and bear (fade at extreme levels) markets via confluence filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1w data for weekly Camarilla pivots (R4/S4 levels)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Camarilla calculation: based on prior week's OHLC
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    weekly_range = 1.1 * (prev_week_high - prev_week_low)
    r4 = prev_week_close + weekly_range * 0.50  # Weekly R4 (strongest resistance)
    s4 = prev_week_close - weekly_range * 0.50  # Weekly S4 (strongest support)
    
    # Align weekly pivots to 6h timeframe (completed weekly bar)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume spike: current volume > 2.0 * 20-period average (moderate threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (34) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Weekly R4/S4 breakout + volume spike + 1d EMA trend alignment
            long_breakout = curr_high > r4_aligned[i]
            short_breakout = curr_low < s4_aligned[i]
            
            # Trend filter: price must be on correct side of 1d EMA
            long_trend = curr_close > ema_34_1d_aligned[i]
            short_trend = curr_close < ema_34_1d_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below weekly R4 (failed breakout) or trend reverses
            if curr_close < r4_aligned[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above weekly S4 (failed breakout) or trend reverses
            if curr_close > s4_aligned[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Confluence_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0