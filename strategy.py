#!/usr/bin/env python3
"""
6h_RelativeStrengthIndex_RSI14_With_WeeklyTrend_Filter
Hypothesis: Use weekly trend direction (price above/below weekly SMA200) to filter RSI14 mean-reversion signals on 6h.
In bull markets (price > weekly SMA200): take RSI < 30 longs, avoid shorts.
In bear markets (price < weekly SMA200): take RSI > 70 shorts, avoid longs.
Weekly trend filter reduces whipsaw by aligning with higher timeframe momentum.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "6h_RelativeStrengthIndex_RSI14_With_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly SMA200
    def sma(arr, period):
        result = np.full_like(arr, np.nan)
        for i in range(period - 1, len(arr)):
            result[i] = np.mean(arr[i - period + 1:i + 1])
        return result
    
    sma200_weekly = sma(close_weekly, 200)
    weekly_trend_up = sma200_weekly > 0  # Valid SMA200 value
    weekly_trend_up = weekly_trend_up & (close_weekly > sma200_weekly)
    weekly_trend_up = np.where(weekly_trend_up, 1, -1)  # 1 for uptrend, -1 for downtrend
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_up)
    
    # Calculate RSI(14) on 6h close
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period + 1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_values = rsi(close, 14)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 AND weekly trend up (bull market bias)
            if rsi_values[i] < 30 and weekly_trend_up_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 AND weekly trend down (bear market bias)
            elif rsi_values[i] > 70 and weekly_trend_up_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) OR weekly trend flips down
            if rsi_values[i] > 50 or weekly_trend_up_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) OR weekly trend flips up
            if rsi_values[i] < 50 or weekly_trend_up_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals