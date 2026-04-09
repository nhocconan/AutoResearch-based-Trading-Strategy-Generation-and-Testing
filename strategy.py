#!/usr/bin/env python3
# 6h_weekly_ichimoku_trend_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d for trend direction and 6h TK cross for entry timing.
# Long when price > 1d Ichimoku cloud and 6h Tenkan > Kijun; short when price < 1d cloud and 6h Tenkan < Kijun.
# Uses weekly higher timeframe filter: only trade when price is above/below weekly Ichimoku cloud.
# Designed to work in both bull (trend following) and bear (cloud acts as dynamic support/resistance) markets.
# Target: 12-30 trades/year (50-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Get 1d data for Ichimoku (primary HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high + period52_low) / 2)
    
    # Align 1d Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        # Fallback to 1d if 1w insufficient
        df_1w = df_1d
        high_1w = high_1d
        low_1w = low_1d
        close_1w = close_1d
    else:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
    
    # Calculate weekly Ichimoku components
    period9_high_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (period9_high_1w + period9_low_1w) / 2
    
    period26_high_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (period26_high_1w + period26_low_1w) / 2
    
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    period52_high_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((period52_high_1w + period52_low_1w) / 2)
    
    # Align weekly Ichimoku components to 6h
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Calculate 6h Tenkan and Kijun for entry timing
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Warmup period: max of all indicator lookbacks
    warmup = max(52, 26, 9)  # Ichimoku needs 52 periods for Senkou B
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(senkou_a_1w_aligned[i]) or 
            np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d cloud boundaries (Senkou Span A/B)
        cloud_top_1d = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Determine 1w cloud boundaries for trend filter
        cloud_top_1w = max(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        cloud_bottom_1w = min(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        # 6h TK cross
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        tk_bullish = tenkan_6h_val > kijun_6h_val
        tk_bearish = tenkan_6h_val < kijun_6h_val
        
        # Price relative to clouds
        price_above_1d_cloud = close[i] > cloud_top_1d
        price_below_1d_cloud = close[i] < cloud_bottom_1d
        price_above_1w_cloud = close[i] > cloud_top_1w
        price_below_1w_cloud = close[i] < cloud_bottom_1w
        
        if position == 1:  # Long position
            # Exit: Price crosses below 1d cloud OR TK cross turns bearish
            if price_below_1d_cloud or tk_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above 1d cloud OR TK cross turns bullish
            if price_above_1d_cloud or tk_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: Price above both clouds AND bullish TK cross
            if (price_above_1d_cloud and price_above_1w_cloud and tk_bullish):
                position = 1
                signals[i] = 0.25
            # Enter short: Price below both clouds AND bearish TK cross
            elif (price_below_1d_cloud and price_below_1w_cloud and tk_bearish):
                position = -1
                signals[i] = -0.25
    
    return signals