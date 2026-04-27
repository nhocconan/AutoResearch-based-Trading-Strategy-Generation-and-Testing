#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud system with 1-day/1-week timeframe filters.
Long when price above cloud + Tenkan > Kijun + weekly bullish bias.
Short when price below cloud + Tenkan < Kijun + weekly bearish bias.
Uses cloud as dynamic support/resistance and TK cross for momentum.
Designed to work in trending markets while avoiding whipsaws in ranges via weekly filter.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

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
    
    # Get 1-day data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou B
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close_1d).shift(26)
    
    # Align Ichimoku components to 6-hour timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou.values)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(25) for trend filter
    close_1w = df_1w['close'].values
    ema_25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_25_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Ichimoku components and weekly EMA
    start_idx = max(26, 26, 26, 26)  # max of lookbacks for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(chikou_6h[i]) or np.isnan(ema_25_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and Ichimoku values
        price_now = close[i]
        tenkan_now = tenkan_6h[i]
        kijun_now = kijun_6h[i]
        senkou_a_now = senkou_a_6h[i]
        senkou_b_now = senkou_b_6h[i]
        chikou_now = chikou_6h[i]
        weekly_trend = ema_25_1w_aligned[i]
        
        # Determine cloud boundaries (Senkou A and B)
        upper_cloud = max(senkou_a_now, senkou_b_now)
        lower_cloud = min(senkou_a_now, senkou_b_now)
        
        # Cloud color: green if Senkou A > Senkou B (bullish), red otherwise
        cloud_bullish = senkou_a_now > senkou_b_now
        
        # Entry conditions
        if position == 0:
            # Long: price above cloud + TK bullish + weekly uptrend
            if (price_now > upper_cloud and 
                tenkan_now > kijun_now and 
                weekly_trend < close[i]):  # Price above weekly EMA = uptrend
                signals[i] = size
                position = 1
            # Short: price below cloud + TK bearish + weekly downtrend
            elif (price_now < lower_cloud and 
                  tenkan_now < kijun_now and 
                  weekly_trend > close[i]):  # Price below weekly EMA = downtrend
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below cloud or TK turns bearish
            if price_now < lower_cloud or tenkan_now < kijun_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above cloud or TK turns bullish
            if price_now > upper_cloud or tenkan_now > kijun_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1wTrend"
timeframe = "6h"
leverage = 1.0