#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d weekly pivot confirmation
# - Uses 1d Ichimoku cloud (Senkou Span A/B) as primary trend filter
# - 6h Tenkan-Kijun cross for entry timing
# - Weekly pivot levels (from 1d data) act as strong support/resistance
# - Price must be above/below weekly pivot to confirm bias
# - Works in all regimes: cloud filters false breaks, pivots provide structure
# - Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Ichimoku and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate weekly pivot points from 1d data (using prior week)
    # Need at least 5 days for prior week
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate pivot for each day based on prior week's OHLC
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().values  # approx
    
    # Standard pivot point: (H + L + C)/3
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    
    # Support and resistance levels
    s1 = (2 * pivot_point) - weekly_high
    r1 = (2 * pivot_point) - weekly_low
    s2 = pivot_point - (weekly_high - weekly_low)
    r2 = pivot_point + (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for all calculations
    start = max(52, 26, 9, 5)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine Ichimoku trend: price above/both cloud lines = bullish
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals
        tk_cross_bull = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bear = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Weekly pivot bias: price above/below pivot
        price_above_pivot = close[i] > pivot_aligned[i]
        price_below_pivot = close[i] < pivot_aligned[i]
        
        if position == 0:
            # Enter long: price above cloud + TK bullish cross + above weekly pivot
            if (price_above_cloud and 
                tk_cross_bull and 
                price_above_pivot):
                position = 1
                signals[i] = position_size
            # Enter short: price below cloud + TK bearish cross + below weekly pivot
            elif (price_below_cloud and 
                  tk_cross_bear and 
                  price_below_pivot):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below cloud OR TK bearish cross
            if (close[i] < cloud_bottom or 
                tk_cross_bear):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above cloud OR TK bullish cross
            if (close[i] > cloud_top or 
                tk_cross_bull):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Ichimoku_WeeklyPivot_v1"
timeframe = "6h"
leverage = 1.0