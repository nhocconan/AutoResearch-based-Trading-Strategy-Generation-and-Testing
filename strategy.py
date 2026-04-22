#!/usr/bin/env python3
"""
Hypothesis: 12-hour Time-Weighted Average Price (TWAP) with 1-day price trend filter.
Long when price > TWAP and 1-day close > 1-day open (bullish daily candle).
Short when price < TWAP and 1-day close < 1-day open (bearish daily candle).
Exit when price crosses TWAP or daily trend reverses.
TWAP provides a fair value reference; daily trend filter ensures alignment with higher timeframe momentum.
Works in both bull and bear markets by following institutional price action while using TWAP for entry timing.
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
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily trend: 1 if bullish (close > open), -1 if bearish (close < open)
    daily_trend = np.where(df_1d['close'] > df_1d['open'], 1, -1)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # TWAP calculation for 12h period (simple average of typical price)
    typical_price = (high + low + close) / 3.0
    
    # Cumulative TWAP (resets daily)
    twap = np.full(n, np.nan)
    cum_tp = 0.0
    count = 0
    
    for i in range(n):
        # Reset at start of each day (00:00 UTC)
        if i > 0 and prices['open_time'].iloc[i].date() != prices['open_time'].iloc[i-1].date():
            cum_tp = 0.0
            count = 0
        
        cum_tp += typical_price[i]
        count += 1
        
        if count > 0:
            twap[i] = cum_tp / count
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(twap[i]) or np.isnan(daily_trend_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above TWAP and daily trend bullish
            if close[i] > twap[i] and daily_trend_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price below TWAP and daily trend bearish
            elif close[i] < twap[i] and daily_trend_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below TWAP or daily trend turns bearish
                if close[i] < twap[i] or daily_trend_aligned[i] == -1:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above TWAP or daily trend turns bullish
                if close[i] > twap[i] or daily_trend_aligned[i] == 1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_TWAP_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0