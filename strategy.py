#!/usr/bin/env python3
"""
6h_WeeklyCamarilla_PivotBreakout_1dEMA50_Trend_VolumeSpike
Hypothesis: Weekly Camarilla pivot breakouts with 1d EMA50 trend filter and volume confirmation on 6h timeframe.
Designed for 12-30 trades/year (50-120 over 4 years) to minimize fee drag.
Uses weekly pivot structure (more significant than daily) for breakout/continuation signals.
Works in bull markets via breakout continuation and bear markets via trend following.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly data for Camarilla pivot calculation (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Prior weekly bar OHLC for Camarilla calculation
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    
    # Weekly Camarilla levels: H4, L4 (strong breakout levels)
    weekly_range = prev_weekly_high - prev_weekly_low
    h4 = prev_weekly_close + weekly_range * 1.1 / 2
    l4 = prev_weekly_close - weekly_range * 1.1 / 2
    
    # Align Weekly Camarilla levels to 6h timeframe (completed weekly bar)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Weekly Camarilla H4/L4 breakout + volume spike + 1d EMA50 trend alignment
            long_breakout = curr_high > h4_aligned[i]
            short_breakout = curr_low < l4_aligned[i]
            
            # Trend filter: price must be on correct side of 1d EMA50
            long_trend = curr_close > ema_50_1d_aligned[i]
            short_trend = curr_close < ema_50_1d_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Weekly Camarilla H4 (failed breakout) 
            # or trend reverses
            if curr_close < h4_aligned[i] or curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Weekly Camarilla L4 (failed breakout) 
            # or trend reverses
            if curr_close > l4_aligned[i] or curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyCamarilla_PivotBreakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0