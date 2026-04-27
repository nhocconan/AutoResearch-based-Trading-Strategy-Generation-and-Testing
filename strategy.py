#!/usr/bin/env python3
"""
1d_WeeklyPivot_Bullish_Engulfing_WeeklyTrend
Hypothesis: On 1d timeframe, bullish engulfing patterns at weekly pivot support/resistance levels with weekly trend filter capture high-probability reversals. Works in bull (bounces from support) and bear (bounces from resistance) markets. Target: 10-20 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot levels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivots to daily timeframe (previous week's levels available at open)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Weekly trend filter: price above/below pivot
    weekly_trend_up = close > pivot_aligned
    weekly_trend_down = close < pivot_aligned
    
    # Bullish engulfing pattern: current candle engulfs previous candle's body
    # Bullish: current close > previous open AND current open < previous close
    bullish_engulf = (close > open_price) & (open_price < np.roll(close, 1)) & (close > np.roll(open_price, 1))
    # Bearish engulfing pattern: current candle engulfs previous candle's body
    # Bearish: current close < previous open AND current open > previous close
    bearish_engulf = (close < open_price) & (open_price > np.roll(close, 1)) & (close < np.roll(open_price, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after first candle to avoid NaN from roll
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish engulfing at or near S1 support with weekly uptrend
            if (bullish_engulf[i] and 
                low[i] <= s1_aligned[i] * 1.02 and  # allow 2% tolerance
                weekly_trend_up[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing at or near R1 resistance with weekly downtrend
            elif (bearish_engulf[i] and 
                  high[i] >= r1_aligned[i] * 0.98 and  # allow 2% tolerance
                  weekly_trend_down[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below pivot or bearish engulfing
            if (close[i] < pivot_aligned[i] or bearish_engulf[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above pivot or bullish engulfing
            if (close[i] > pivot_aligned[i] or bullish_engulf[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_Bullish_Engulfing_WeeklyTrend"
timeframe = "1d"
leverage = 1.0