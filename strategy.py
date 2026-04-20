#!/usr/bin/env python3
"""
6h_1d_ADX_SuperTrend_Filter
Hypothesis: Use 1-day ADX to filter trending vs ranging markets. In trending markets (ADX>25), 
use SuperTrend(10,3) on 6h for trend following. In ranging markets (ADX<=25), fade at 1-day 
support/resistance levels. This adapts to market regime to work in both bull and bear markets.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
"""

name = "6h_1d_ADX_SuperTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX with proper smoothing"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        
        # True Range
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[1:period])  # First average
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Handle first values
    tr[0] = high[0] - low[0]
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / atr
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr
    
    dx = np.zeros_like(high)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    
    return adx

def supertrend(high, low, close, atr_period=10, multiplier=3):
    """Calculate SuperTrend indicator"""
    # Calculate ATR
    tr = np.zeros_like(high)
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    tr[0] = high[0] - low[0]
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate basic upper and lower bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Initialize final bands
    final_ub = np.zeros_like(high)
    final_lb = np.zeros_like(high)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    # Calculate final bands
    for i in range(1, len(high)):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Determine trend
    supertrend = np.zeros_like(high)
    trend = np.ones_like(high)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = final_lb[0]
    trend[0] = 1
    
    for i in range(1, len(high)):
        if close[i] > final_ub[i-1]:
            trend[i] = 1
        elif close[i] < final_lb[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
        if trend[i] == 1:
            supertrend[i] = final_lb[i]
        else:
            supertrend[i] = final_ub[i]
    
    return supertrend, trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ADX for regime detection
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate SuperTrend on 6h data
    st, st_trend = supertrend(high, low, close, 10, 3)
    
    # Calculate 1-day support/resistance levels (using pivots)
    # Simple approach: use recent highs/lows as S/R
    lookback = 10
    resistance = np.full_like(close, np.nan)
    support = np.full_like(close, np.nan)
    
    for i in range(lookback, len(high)):
        resistance[i] = np.max(high[i-lookback:i])
        support[i] = np.min(low[i-lookback:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(st[i]) or np.isnan(st_trend[i]) or
            np.isnan(close[i]) or np.isnan(resistance[i]) or np.isnan(support[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine market regime based on 1-day ADX
            if adx_1d_aligned[i] > 25:  # Trending market
                # Follow SuperTrend direction
                if st_trend[i] == 1:  # Uptrend
                    signals[i] = 0.25
                    position = 1
                else:  # Downtrend
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market
                # Fade at support/resistance levels
                if close[i] <= support[i] * 1.001:  # Near support, go long
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= resistance[i] * 0.999:  # Near resistance, go short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if adx_1d_aligned[i] > 25:  # Trending market
                # Exit when SuperTrend turns bearish
                if st_trend[i] == -1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging market
                # Exit when price reaches resistance or stops near support
                if close[i] >= resistance[i] * 0.999:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if adx_1d_aligned[i] > 25:  # Trending market
                # Exit when SuperTrend turns bullish
                if st_trend[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging market
                # Exit when price reaches support or stops near resistance
                if close[i] <= support[i] * 1.001:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals