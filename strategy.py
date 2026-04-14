#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Ichimoku Cloud (Tenkan/Kijun/Senkou Span) as trend filter and 6h price position relative to cloud for entry.
# Ichimoku Cloud from weekly timeframe provides strong trend identification - price above cloud = bullish trend, below = bearish.
# Entry: Go long when price crosses above cloud base (Senkou Span B) in bullish trend, short when crosses below in bearish trend.
# Exit: Reverse when price crosses the opposite cloud boundary (Tenkan-Kijun midpoint).
# This captures major trends while avoiding whipsaws in ranging markets. Weekly filter ensures we only trade with the dominant trend.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE for Ichimoku Cloud calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need ~1 year of weekly data
        return np.zeros(n)
    
    # Calculate Ichimoku Cloud components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted forward 52 periods
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Calculate cloud boundaries (shifted forward by 26 periods for Senkou Span A/B)
    # Since we aligned the already-shifted spans, we use them directly
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Middle line for exit (Tenkan-Kijun midpoint)
    cloud_middle = (tenkan_aligned + kijun_aligned) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 52  # Need Senkou Span B calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or
            np.isnan(cloud_middle[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for cloud breakouts in direction of trend
            # Bullish trend: price above cloud
            # Bearish trend: price below cloud
            
            # Long: price breaks above cloud top AND bullish trend (price above cloud)
            if (close[i] > cloud_top[i] and 
                close[i] > cloud_bottom[i]):  # Price above cloud (bullish)
                position = 1
                signals[i] = position_size
            # Short: price breaks below cloud bottom AND bearish trend (price below cloud)
            elif (close[i] < cloud_bottom[i] and 
                  close[i] < cloud_top[i]):  # Price below cloud (bearish)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to cloud middle (Tenkan-Kijun midpoint)
            if close[i] <= cloud_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to cloud middle
            if close[i] >= cloud_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wIchimoku_Cloud_Breakout_MiddleExit_v1"
timeframe = "6h"
leverage = 1.0