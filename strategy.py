#!/usr/bin/env python3
"""
6h_1d_1w_ichimoku_cloud_filter_v1
Strategy: 6h Ichimoku Cloud breakout with 1d/1w trend alignment
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses Ichimoku Cloud (Tenkan/Kijun/Senkou) on 6h for breakout signals, filtered by 1d and 1w trend direction (price above/below Kumo). The cloud acts as dynamic support/resistance. In bull markets, longs when price breaks above cloud with 1d/1w uptrend; in bear markets, shorts when price breaks below cloud with 1d/1w downtrend. Avoids false breakouts in chop by requiring multi-timeframe alignment. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_ichimoku_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 6h Ichimoku Cloud components
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
    
    # Kumo (Cloud) boundaries: Senkou A and B shifted forward 26 periods
    # For backtesting, we use current Senkou spans as cloud boundaries
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # 1d trend: price above/below 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w trend: price above/below 1w EMA20
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters: price above/below higher timeframe EMAs
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        uptrend_1w = price_close > ema_20_1w_aligned[i]
        downtrend_1w = price_close < ema_20_1w_aligned[i]
        
        # Cloud breakout conditions
        breakout_above_cloud = price_close > upper_cloud[i-1] and close[i-1] <= upper_cloud[i-1]
        breakout_below_cloud = price_close < lower_cloud[i-1] and close[i-1] >= lower_cloud[i-1]
        
        # Long: breakout above cloud with 1d/1w uptrend
        long_signal = breakout_above_cloud and uptrend_1d and uptrend_1w
        
        # Short: breakout below cloud with 1d/1w downtrend
        short_signal = breakout_below_cloud and downtrend_1d and downtrend_1w
        
        # Exit when price re-enters the cloud
        exit_long = position == 1 and price_close < lower_cloud[i]
        exit_short = position == -1 and price_close > upper_cloud[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals