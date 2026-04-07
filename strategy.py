#!/usr/bin/env python3
"""
6H Ichimoku Cloud with 1D Filter
Hypothesis: Ichimoku TK cross with cloud filter from daily timeframe provides high-probability entries. 
Only take longs when price > daily cloud and TK crosses up; shorts when price < daily cloud and TK crosses down.
Uses 6h timeframe to balance signal frequency and quality. Designed to work in both bull and bear markets
by requiring alignment with higher timeframe trend (daily cloud). Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Get 1d Ichimoku cloud for filter
    # Daily Tenkan and Kijun
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_tenkan = (pd.Series(d_high).rolling(window=9, min_periods=9).max().values + 
                pd.Series(d_low).rolling(window=9, min_periods=9).min().values) / 2
    d_kijun = (pd.Series(d_high).rolling(window=26, min_periods=26).max().values + 
               pd.Series(d_low).rolling(window=26, min_periods=26).min().values) / 2
    d_senkou_a = ((d_tenkan + d_kijun) / 2)
    d_senkou_b = ((pd.Series(d_high).rolling(window=52, min_periods=52).max().values + 
                   pd.Series(d_low).rolling(window=52, min_periods=52).min().values) / 2)
    
    # Align daily Ichimoku components to 6h
    d_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_a)
    d_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_b)
    
    # Determine if price is above or below daily cloud
    # When Senkou A > Senkou B, cloud is green (bullish); else red (bearish)
    # For filter: price above cloud = bullish bias, price below cloud = bearish bias
    daily_cloud_top = np.maximum(d_senkou_a_aligned, d_senkou_b_aligned)
    daily_cloud_bottom = np.minimum(d_senkou_a_aligned, d_senkou_b_aligned)
    price_above_daily_cloud = close > daily_cloud_top
    price_below_daily_cloud = close < daily_cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start loop after enough data for Ichimoku (52 periods for Senkou B)
    for i in range(52, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(price_above_daily_cloud[i]) or np.isnan(price_below_daily_cloud[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TK cross down OR price closes below daily cloud
            if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or not price_above_daily_cloud[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross up OR price closes above daily cloud
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or not price_below_daily_cloud[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: TK cross up AND price above daily cloud
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) and price_above_daily_cloud[i]:
                position = 1
                signals[i] = 0.25
            # Short: TK cross down AND price below daily cloud
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) and price_below_daily_cloud[i]:
                position = -1
                signals[i] = -0.25
    
    return signals