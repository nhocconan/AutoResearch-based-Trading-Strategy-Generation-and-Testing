#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrend_VolumeConfirm
Hypothesis: On 12h timeframe, Camarilla H3/L3 breakouts from prior 1d combined with 1d EMA50 trend filter and volume confirmation.
Camarilla H3/L3 levels represent strong intraday support/resistance derived from prior day's range. 
Breakouts with volume confirmation and trend alignment capture institutional participation. 
1d EMA50 ensures alignment with intermediate trend to avoid counter-trend whipsaws.
Designed for low trade frequency (12-37/year) to minimize fee drag while maintaining edge in both bull and bear markets via confluence filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and EMA trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Prior 1d OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3/L3 (strong intraday resistance/support)
    range_ = prev_high - prev_low
    h3 = prev_close + range_ * 1.1 / 4   # H3 level
    l3 = prev_close - range_ * 1.1 / 4   # L3 level
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0 * 20-period average (moderate threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (50) + volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: H3/L3 breakout + volume spike + 1d EMA trend alignment
            long_breakout = curr_high > h3_aligned[i]
            short_breakout = curr_low < l3_aligned[i]
            
            # Trend filter: price must be on correct side of 1d EMA50
            long_trend = curr_close > ema_50_1d_aligned[i]
            short_trend = curr_close < ema_50_1d_aligned[i]
            
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
            # Long position: exit when price closes below H3 (failed breakout) or trend reverses
            if curr_close < h3_aligned[i] or curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L3 (failed breakout) or trend reverses
            if curr_close > l3_aligned[i] or curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0