#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK cross and 1d weekly pivot direction filter
# Uses Ichimoku (Tenkan/Kijun/Senkou) from 6h for trend and momentum signals
# Only takes trades aligned with 1d weekly pivot (price above weekly pivot = long bias, below = short bias)
# Weekly pivot calculated from prior 1d week (Mon-Sun) high/low/close
# Position size 0.25 to manage drawdown, target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: weekly pivot provides structural bias, Ichimoku TK cross captures momentum

name = "6h_1d_ichimoku_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    for i in range(n):
        if i < 9:
            period9_high[i] = np.nan
            period9_low[i] = np.nan
        else:
            period9_high[i] = np.max(high[i-9:i])
            period9_low[i] = np.min(low[i-9:i])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    for i in range(n):
        if i < 26:
            period26_high[i] = np.nan
            period26_low[i] = np.nan
        else:
            period26_high[i] = np.max(high[i-26:i])
            period26_low[i] = np.min(low[i-26:i])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    for i in range(n):
        if i < 52:
            period52_high[i] = np.nan
            period52_low[i] = np.nan
        else:
            period52_high[i] = np.max(high[i-52:i])
            period52_low[i] = np.min(low[i-52:i])
    senkou_b = (period52_high + period52_low) / 2
    
    # Calculate weekly pivot from 1d data (prior week's high, low, close)
    # Assuming 1d data is daily, weekly pivot uses prior 7 days (Mon-Sun)
    weekly_high = np.full(len(df_1d), np.nan)
    weekly_low = np.full(len(df_1d), np.nan)
    weekly_close = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 7:
            weekly_high[i] = np.nan
            weekly_low[i] = np.nan
            weekly_close[i] = np.nan
        else:
            weekly_high[i] = np.max(df_1d['high'].iloc[i-7:i])
            weekly_low[i] = np.min(df_1d['low'].iloc[i-7:i])
            weekly_close[i] = df_1d['close'].iloc[i-1]  # prior day's close as weekly close proxy
    
    # Weekly pivot point: (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align 1d weekly pivot to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup for longest Ichimoku component
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(weekly_pivot_6h[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku TK cross: Tenkan crosses above/below Kijun
        tk_cross_above = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_below = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price relative to cloud: above cloud = bullish, below cloud = bearish
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Weekly pivot bias: price above pivot = long bias, below = short bias
        price_above_pivot = close[i] > weekly_pivot_6h[i]
        price_below_pivot = close[i] < weekly_pivot_6h[i]
        
        if position == 1:  # Long position
            # Exit: TK cross below OR price breaks below cloud OR pivot bias turns bearish
            if tk_cross_below or price_below_cloud or (not price_above_pivot and close[i] < weekly_pivot_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross above OR price breaks above cloud OR pivot bias turns bullish
            if tk_cross_above or price_above_cloud or (not price_below_pivot and close[i] > weekly_pivot_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry: TK cross in direction of price vs cloud AND aligned with weekly pivot bias
            if tk_cross_above and price_above_cloud and price_above_pivot:
                position = 1
                signals[i] = 0.25
            elif tk_cross_below and price_below_cloud and price_below_pivot:
                position = -1
                signals[i] = -0.25
    
    return signals