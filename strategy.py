#!/usr/bin/env python3
"""
12h_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Weekly Camarilla H4/L4 levels act as strong support/resistance on 12h timeframe.
Breakouts above H4 or below L4 with volume spike and weekly EMA50 trend alignment capture
institutional moves. Designed for 50-150 trades over 4 years on 12h timeframe.
Works in both bull and bear markets via weekly EMA50 trend filter (only trade in trend direction).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    # Camarilla equations
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    # Resistance levels
    r1 = close + (range_hl * 1.1 / 12)
    r2 = close + (range_hl * 1.1 / 6)
    r3 = close + (range_hl * 1.1 / 4)
    r4 = close + (range_hl * 1.1 / 2)
    
    # Support levels
    s1 = close - (range_hl * 1.1 / 12)
    s2 = close - (range_hl * 1.1 / 6)
    s3 = close - (range_hl * 1.1 / 4)
    s4 = close - (range_hl * 1.1 / 2)
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA50 trend filter and Camarilla levels (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly Camarilla levels (H4/L4) for 12h breakout signals
    h4, _, _, _, _, _, _, l4 = calculate_camarilla(
        df_1w['high'].values, 
        df_1w['low'].values, 
        df_1w['close'].values
    )
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (50) + volume MA (20)
    start_idx = max(50, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume spike + weekly EMA50 trend alignment
            long_entry = (curr_close > h4_aligned[i]) and vol_ma[i] > 0 and volume_spike[i] and (curr_close > ema_50_1w_aligned[i])
            short_entry = (curr_close < l4_aligned[i]) and vol_ma[i] > 0 and volume_spike[i] and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below H4 or trend turns bearish
            if curr_close < h4_aligned[i] or curr_close < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L4 or trend turns bullish
            if curr_close > l4_aligned[i] or curr_close > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0