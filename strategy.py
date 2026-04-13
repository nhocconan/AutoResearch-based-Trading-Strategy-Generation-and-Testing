#!/usr/bin/env python3
"""
1d_1w_CamarillaPivot_Breakout_TrendFilter_v1
Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance. 
Breakouts above R4 or below S4 with weekly ADX trend filter capture strong momentum moves. 
This strategy works in both bull and bear markets by following strong trends only. 
Target: 10-20 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    R4 = close_1d + (range_hl * 1.1 / 2)
    R3 = close_1d + (range_hl * 1.1 / 4)
    R2 = close_1d + (range_hl * 1.1 / 6)
    R1 = close_1d + (range_hl * 1.1 / 12)
    S1 = close_1d - (range_hl * 1.1 / 12)
    S2 = close_1d - (range_hl * 1.1 / 6)
    S3 = close_1d - (range_hl * 1.1 / 4)
    S4 = close_1d - (range_hl * 1.1 / 2)
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period)
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    up_move = np.where(high_1w - np.roll(high_1w, 1) > 0, high_1w - np.roll(high_1w, 1), 0)
    down_move = np.where(np.roll(low_1w, 1) - low_1w > 0, np.roll(low_1w, 1) - low_1w, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    period = 14
    atr = wilders_smooth(tr, period)
    plus_dm = wilders_smooth(up_move, period)
    minus_dm = wilders_smooth(down_move, period)
    
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    strong_trend = adx > 25  # Strong trend filter
    
    # Align all signals to daily timeframe (since price is daily)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(2, n):  # Start from 2 to ensure we have previous day's pivots
        # Skip if data not ready
        if (np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Use previous day's pivots for breakout (avoid look-ahead)
        R4_prev = R4_aligned[i-1]
        S4_prev = S4_aligned[i-1]
        
        # Entry conditions: Break of R4 or S4 with strong weekly trend
        long_break = close[i] > R4_prev
        short_break = close[i] < S4_prev
        
        long_entry = long_break and strong_trend_aligned[i] > 0.5
        short_entry = short_break and strong_trend_aligned[i] > 0.5
        
        # Exit when price returns to previous day's close (mean reversion to equilibrium)
        prev_close = close_1d[-1] if i-1 < len(close_1d) else close[i-1]
        # Simplified: exit when price crosses the previous day's close
        exit_long = position == 1 and close[i] < close[i-1]
        exit_short = position == -1 and close[i] > close[i-1]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_CamarillaPivot_Breakout_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0