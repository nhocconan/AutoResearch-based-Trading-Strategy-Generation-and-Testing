#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with weekly trend filter
# Enter long when TK crosses above Kijun (Tenkan > Kijun) and price above Kumo (cloud) from 1d
# Enter short when TK crosses below Kijun and price below Kumo from 1d
# Uses weekly trend filter: only long when price > weekly EMA50, short when price < weekly EMA50
# Designed to capture trend continuation in both bull and bear markets with strict entry filters
# Target: 50-150 total trades over 4 years

name = "6h_ichimoku_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52 periods) on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    # For simplicity, we use current Senkou spans as cloud boundaries
    # In practice, cloud is plotted 26 periods ahead, but we use current for signal
    # Upper cloud = max(senkou_a, senkou_b), Lower cloud = min(senkou_a, senkou_b)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # 1d EMA50 for trend filter (daily trend)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly EMA50 for stronger trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: TK cross below KJ OR price below cloud OR weekly trend breaks
            if (tenkan[i] < kijun[i] or close[i] < lower_cloud[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross above KJ OR price above cloud OR weekly trend breaks
            if (tenkan[i] > kijun[i] or close[i] > upper_cloud[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross with price outside cloud and weekly trend alignment
            # Bullish: TK crosses above KJ, price above cloud, above weekly EMA50
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # TK cross above KJ
                close[i] > upper_cloud[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: TK crosses below KJ, price below cloud, below weekly EMA50
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # TK cross below KJ
                  close[i] < lower_cloud[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals