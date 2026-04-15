#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Calculate 12h EMA(50) for trend
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current 4h volume > 1.8x 20-period average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # Trend filter: price above/below 12h EMA50
    trend_filter_long = close > ema_12h_aligned
    trend_filter_short = close < ema_12h_aligned
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long conditions: price breaks above R1 with uptrend
            if close[i] > r1_aligned[i] and trend_filter_long[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below S1 with downtrend
            elif close[i] < s1_aligned[i] and trend_filter_short[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Flat when conditions not met
        else:
            signals[i] = 0.0  # Flat when volume filter fails
    
    return signals

name = "4h_DailyPivot_R1_S1_Breakout_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0