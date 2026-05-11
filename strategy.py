#/usr/bin/env python3
"""
6H_WEEKLY_PIVOT_TREND_FOLLOW
Hypothesis: Price tends to continue in the direction of the previous weekly trend after breaking
key weekly pivot levels (CP, R1, S1) on the 6h chart. Uses weekly pivot points as dynamic support/resistance
and 1d EMA for trend filter. Works in both bull and bear markets by following the higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) with low frequency to minimize fee drag.
"""

name = "6H_WEEKLY_PIVOT_TREND_FOLLOW"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Weekly Pivot Points (using previous week's OHLC) ---
    # Calculate pivot points from previous week's data
    # We need to shift the weekly data by 1 to use only completed weeks
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    pp = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3.0
    r1 = 2 * pp - np.roll(low_1w, 1)
    s1 = 2 * pp - np.roll(high_1w, 1)
    r2 = pp + (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    s2 = pp - (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    
    # First week has no previous week, so set to NaN
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    r2[0] = np.nan
    s2[0] = np.nan
    
    # Align weekly pivot points to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                # Hold current position until exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions: price breaks weekly pivot levels in direction of daily trend
        # Long: price crosses above R1 and daily EMA34 is rising (trending up)
        # Short: price crosses below S1 and daily EMA34 is falling (trending down)
        long_breakout = (close[i] > r1_aligned[i]) and (close[i-1] <= r1_aligned[i-1]) and (ema_34_aligned[i] > ema_34_aligned[i-1])
        short_breakout = (close[i] < s1_aligned[i]) and (close[i-1] >= s1_aligned[i-1]) and (ema_34_aligned[i] < ema_34_aligned[i-1])
        
        # Exit conditions: price reverses back through pivot point or trend changes
        long_exit = (close[i] < pp_aligned[i]) or (ema_34_aligned[i] < ema_34_aligned[i-1])
        short_exit = (close[i] > pp_aligned[i]) or (ema_34_aligned[i] > ema_34_aligned[i-1])
        
        if position == 0:
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                if long_exit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if short_exit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals