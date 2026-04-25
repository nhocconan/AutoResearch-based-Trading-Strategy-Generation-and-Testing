#!/usr/bin/env python3
"""
6h_Ichimoku_Kijun_Bounce_1dTrendFilter_v1
Hypothesis: Trade price bounces off the Ichimoku Kijun-Sen (base line) on 6h timeframe, 
filtered by 1d EMA50 trend direction. In bull markets, buy dips to Kijun-Sen; in bear markets, 
sell rallies to Kijun-Sen. The Kijun-Sen acts as dynamic support/resistance, and 
aligning with higher timeframe trend increases win rate. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for longest indicator (Senkou Span B = 52)
    start_idx = period_senkou_b
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Determine if price is above/below Ichimoku cloud
        above_cloud = (close[i] > senkou_span_a[i]) and (close[i] > senkou_span_b[i])
        below_cloud = (close[i] < senkou_span_a[i]) and (close[i] < senkou_span_b[i])
        
        if position == 0:
            # Long setup: price bounces up from Kijun-sen in bullish trend AND above cloud
            long_setup = (close[i] > kijun_sen[i]) and (close[i-1] <= kijun_sen[i-1]) and htf_1d_bullish and above_cloud
            
            # Short setup: price bounces down from Kijun-sen in bearish trend AND below cloud
            short_setup = (close[i] < kijun_sen[i]) and (close[i-1] >= kijun_sen[i-1]) and htf_1d_bearish and below_cloud
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price crosses below Kijun-sen (stop) OR trend turns bearish OR price falls below cloud
            if (close[i] < kijun_sen[i]) or (not htf_1d_bullish) or (not above_cloud):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses above Kijun-sen (stop) OR trend turns bullish OR price rises above cloud
            if (close[i] > kijun_sen[i]) or (htf_1d_bullish) or (above_cloud):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kijun_Bounce_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0