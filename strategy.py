#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_1wTrend_Filter
Hypothesis: Long/short at weekly pivot point breakouts with weekly trend filter on daily timeframe.
Weekly pivots act as institutional support/resistance. Trend filter avoids counter-trend trades.
Targets 10-25 trades/year by requiring both pivot breakout and weekly trend alignment.
Works in bull markets (breakouts with trend) and bear markets (mean reversion at pivots in range).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot points and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Using previous week's OHLC
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values
    prev_weekly_open = df_weekly['open'].shift(1).values
    
    # Pivot point (P) = (H + L + C) / 3
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    
    # Support and resistance levels
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    r1 = 2 * pivot - prev_weekly_low
    s1 = 2 * pivot - prev_weekly_high
    r2 = pivot + (prev_weekly_high - prev_weekly_low)
    s2 = pivot - (prev_weekly_high - prev_weekly_low)
    
    # Weekly trend: price > weekly close = uptrend, < weekly close = downtrend
    weekly_trend_up = prev_weekly_close > prev_weekly_open  # bullish weekly candle
    weekly_trend_down = prev_weekly_close < prev_weekly_open  # bearish weekly candle
    
    # Align weekly data to daily
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: price breaks above R1 with weekly uptrend OR price holds above S2 with weekly uptrend
        long_breakout = (close[i] > r1_aligned[i] and weekly_trend_up_aligned[i] > 0.5)
        long_support = (close[i] > s2_aligned[i] and weekly_trend_up_aligned[i] > 0.5 and 
                       close[i-1] <= s2_aligned[i-1] if i > 0 else False)
        
        # Short: price breaks below S1 with weekly downtrend OR price fails at R2 with weekly downtrend
        short_breakdown = (close[i] < s1_aligned[i] and weekly_trend_down_aligned[i] > 0.5)
        short_resistance = (close[i] < r2_aligned[i] and weekly_trend_down_aligned[i] > 0.5 and
                           close[i-1] >= r2_aligned[i-1] if i > 0 else False)
        
        # Exit conditions
        long_exit = close[i] < pivot_aligned[i]  # Exit long when price falls below pivot
        short_exit = close[i] > pivot_aligned[i]  # Exit short when price rises above pivot
        
        if (long_breakout or long_support) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_breakdown or short_resistance) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Close long
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.25   # Close short
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivot_Breakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0