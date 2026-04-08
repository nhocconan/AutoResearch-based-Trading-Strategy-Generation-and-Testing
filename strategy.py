#!/usr/bin/env python3
# 6h_ichimoku_1d_trend_follow
# Hypothesis: Ichimoku cloud filter from daily timeframe combined with Tenkan/Kijun crossover on 6h.
# Long when price is above daily Ichimoku cloud AND Tenkan crosses above Kijun on 6h.
# Short when price is below daily Ichimoku cloud AND Tenkan crosses below Kijun on 6h.
# Uses daily Ichimoku for trend filter (avoiding counter-trend trades) and 6h for entry timing.
# Target: 12-30 trades/year with strict confluence to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily Ichimoku data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for cloud)
    
    # Align Ichimoku components to 6h timeframe (with proper delay for forward-looking elements)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Calculate 6h Tenkan/Kijun for crossover signals
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h_cross = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    
    max_high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h_cross = (max_high_kijun_6h + min_low_kijun_6h) / 2
    
    # Determine trend from daily Ichimoku cloud
    # Price above cloud: bullish, Price below cloud: bearish
    above_cloud = (close > senkou_a_6h) & (close > senkou_b_6h)
    below_cloud = (close < senkou_a_6h) & (close < senkou_b_6h)
    
    # Calculate Tenkan/Kijun crossover on 6h
    tenkan_cross_above = (tenkan_6h_cross > kijun_6h_cross) & (tenkan_6h_cross <= kijun_6h_cross + 1e-10)  # Avoid exact equality issues
    tenkan_cross_below = (tenkan_6h_cross < kijun_6h_cross) & (tenkan_6h_cross >= kijun_6h_cross - 1e-10)
    
    # More reliable crossover detection using previous bar
    tenkan_prev = np.roll(tenkan_6h_cross, 1)
    kijun_prev = np.roll(kijun_6h_cross, 1)
    tenkan_prev[0] = tenkan_6h_cross[0]
    kijun_prev[0] = kijun_6h_cross[0]
    
    tenkan_cross_above = (tenkan_6h_cross > kijun_6h_cross) & (tenkan_prev <= kijun_prev)
    tenkan_cross_below = (tenkan_6h_cross < kijun_6h_cross) & (tenkan_prev >= kijun_prev)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(26, 52) + 10  # Ensure sufficient data for all components
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(tenkan_6h_cross[i]) or np.isnan(kijun_6h_cross[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below cloud OR Tenkan crosses below Kijun
            if below_cloud[i] or tenkan_cross_below[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above cloud OR Tenkan crosses above Kijun
            if above_cloud[i] or tenkan_cross_above[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above cloud AND Tenkan crosses above Kijun
            if above_cloud[i] and tenkan_cross_above[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud AND Tenkan crosses below Kijun
            elif below_cloud[i] and tenkan_cross_below[i]:
                position = -1
                signals[i] = -0.25
    
    return signals