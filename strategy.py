#!/usr/bin/env python3
"""
12h_WeeklyCamarilla_H4L4_Breakout_1wTrend_VolumeConfirm
Hypothesis: On 12h timeframe, weekly Camarilla pivot breakouts (H4/L4 levels) combined with 1w EMA trend filter and volume confirmation.
Weekly pivots capture major support/resistance from prior week, reducing false breakouts. 1w EMA ensures alignment with long-term trend.
Volume spike confirms institutional participation. Designed for 12h timeframe to capture fewer, higher-quality trades (12-37/year).
Works in both bull (breakout continuation) and bear (fade at extreme levels) markets via confluence filtering.
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
    
    # 1w data for EMA trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 trend filter
    ema_34_1w = calculate_ema(df_1w['close'].values, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1w data for weekly Camarilla pivots (H4/L4 levels)
    # Weekly Camarilla calculation: based on prior week's OHLC
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    weekly_range = prev_week_high - prev_week_low
    h4 = prev_week_close + weekly_range * 0.55  # Weekly H4 (strong resistance)
    l4 = prev_week_close - weekly_range * 0.55  # Weekly L4 (strong support)
    
    # Align weekly pivots to 12h timeframe (completed weekly bar)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Volume spike: current volume > 2.5 * 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (34) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Weekly H4/L4 breakout + volume spike + 1w EMA trend alignment
            long_breakout = curr_high > h4_aligned[i]
            short_breakout = curr_low < l4_aligned[i]
            
            # Trend filter: price must be on correct side of 1w EMA
            long_trend = curr_close > ema_34_1w_aligned[i]
            short_trend = curr_close < ema_34_1w_aligned[i]
            
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
            # Long position: exit when price closes below weekly H4 (failed breakout) or trend reverses
            if curr_close < h4_aligned[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above weekly L4 (failed breakout) or trend reverses
            if curr_close > l4_aligned[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyCamarilla_H4L4_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0