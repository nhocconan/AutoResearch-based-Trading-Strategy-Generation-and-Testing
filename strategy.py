# 6h_WeeklyPivot_DailyTrend_Signal
# Hypothesis: For 6h timeframe, trade in direction of 1d EMA34 trend only when price breaks weekly pivot S1/R1.
# Use weekly pivot from 1w timeframe for structure and 1d EMA34 for trend filter.
# Enter long when price breaks above weekly R1 and 1d EMA34 rising; short when breaks below weekly S1 and 1d EMA34 falling.
# Exit when price crosses 1d EMA34 (trend change) or reverses to weekly pivot point.
# Designed for 6h timeframe targeting 15-30 trades/year with trend-following edge in both bull and bear markets.

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
    
    # Load weekly data for pivot points (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Load daily data for EMA34 trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 and daily EMA34 is rising (trend up)
            if (close[i] > r1_aligned[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 and daily EMA34 is falling (trend down)
            elif (close[i] < s1_aligned[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal (EMA34 flatten/reverse) or price returns to weekly pivot
            if position == 1:
                # Exit on trend reversal or return to pivot
                if (ema34_1d_aligned[i] <= ema34_1d_aligned[i-1] or 
                    close[i] <= pivot_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on trend reversal or return to pivot
                if (ema34_1d_aligned[i] >= ema34_1d_aligned[i-1] or 
                    close[i] >= pivot_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DailyTrend_Signal"
timeframe = "6h"
leverage = 1.0