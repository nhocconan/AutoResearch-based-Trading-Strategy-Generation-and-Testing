#!/usr/bin/env python3
"""
4h_Camarilla_H4L4_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: On 4h timeframe, Camarilla H4/L4 breakouts with 12h EMA50 trend filter and volume spike.
Uses H4/L4 levels (wider than H3/L3) for fewer, higher-quality breakouts. Volume spike confirms institutional participation.
12h EMA50 trend filter ensures trades align with higher timeframe momentum. Works in bull (breakout continuation) 
and bear (mean reversion at H4/L4) markets. Target: 20-50 trades/year to stay within proven winning range for 4h.
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
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA50 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50_12h = calculate_ema(df_12h['close'].values, 50)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d data for Camarilla pivots (H4/L4 levels - wider bands for fewer trades)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla levels: H4, L4 (widest bands = fewest trades)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_range = 1.1 * (prev_high - prev_low)
    h4 = prev_close + camarilla_range * 0.50  # H4 level (widest)
    l4 = prev_close - camarilla_range * 0.50  # L4 level (widest)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (50) + volume MA (20) + Camarilla (2)
    start_idx = max(50, 20, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H4/L4 breakout + volume spike + 12h EMA50 trend alignment
            long_breakout = curr_high > h4_aligned[i]
            short_breakout = curr_low < l4_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and 
                         (curr_close > ema_50_12h_aligned[i]))
            short_entry = (short_breakout and volume_spike[i] and 
                          (curr_close < ema_50_12h_aligned[i]))
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below H4 (failed breakout) or trend turns bearish
            if curr_close < h4_aligned[i] or curr_close < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L4 (failed breakout) or trend turns bullish
            if curr_close > l4_aligned[i] or curr_close > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0