#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: On daily timeframe, Camarilla H3/L3 levels act as strong support/resistance.
Breakouts above H3 or below L3 with volume spike and 1-week EMA50 trend alignment capture
institutional moves. Works in bull/bear via 1w EMA50 trend filter (only trade in trend direction).
Designed for 30-100 trades over 4 years on 1d timeframe (~7-25/year).
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
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 trend filter
    ema_50_1w = calculate_ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Camarilla levels from 1w data (for 1d breakout signals)
    # We use 1w high/low/close to calculate Camarilla levels for the current 1d bar
    h3, l3, _, _, _, _, _, _ = calculate_camarilla(
        df_1w['high'].values, 
        df_1w['low'].values, 
        df_1w['close'].values
    )[:2]  # We only need H3 and L3 for this strategy
    # Recalculate to get all levels properly
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
        df_1w['high'].values, 
        df_1w['low'].values, 
        df_1w['close'].values
    )
    h3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (50) + volume MA (20)
    start_idx = max(50, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume spike + 1w EMA50 trend alignment
            long_entry = (curr_close > h3_aligned[i]) and vol_ma[i] > 0 and volume_spike[i] and (curr_close > ema_50_1w_aligned[i])
            short_entry = (curr_close < l3_aligned[i]) and vol_ma[i] > 0 and volume_spike[i] and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below H3 or trend turns bearish
            if curr_close < h3_aligned[i] or curr_close < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L3 or trend turns bullish
            if curr_close > l3_aligned[i] or curr_close > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0