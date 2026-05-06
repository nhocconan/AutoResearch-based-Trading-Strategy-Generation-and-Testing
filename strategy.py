#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Ichimoku Cloud filter from 1-day timeframe for trend direction,
# with entry triggered by Tenkan/Kijun cross on 1-day and confirmed by price position relative to cloud.
# Long when price is above cloud, Tenkan > Kijun, and price closes above Tenkan.
# Short when price is below cloud, Tenkan < Kijun, and price closes below Tenkan.
# Uses weekly timeframe for regime filter (price above/below weekly cloud) to avoid counter-trend trades.
# Designed to work in trending markets (both bull and bear) by following Ichimoku trend signals.
# Target: 50-150 total trades over 4 years = 12-37/year with 0.25 position sizing.

name = "6h_Ichimoku_Cloud_Filter_TK_Cross_1d"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for Ichimoku calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1-day data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Get 1-week data for regime filter (cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Ichimoku components
    tenkan_sen_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2)
    senkou_span_b_1w = (pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                        pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2
    
    # Align weekly Ichimoku components to 6h timeframe
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w.values)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w.values)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(senkou_span_a_1w_aligned[i]) or np.isnan(senkou_span_b_1w_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is above or below weekly cloud (regime filter)
        weekly_cloud_top = np.maximum(senkou_span_a_1w_aligned[i], senkou_span_b_1w_aligned[i])
        weekly_cloud_bottom = np.minimum(senkou_span_a_1w_aligned[i], senkou_span_b_1w_aligned[i])
        price_above_weekly_cloud = close[i] > weekly_cloud_top
        price_below_weekly_cloud = close[i] < weekly_cloud_bottom
        
        if position == 0:
            # Long: price above daily cloud, Tenkan > Kijun, price closes above Tenkan
            daily_cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
            daily_cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
            price_above_daily_cloud = close[i] > daily_cloud_top
            tenkan_above_kijun = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            price_above_tenkan = close[i] > tenkan_sen_aligned[i]
            
            if (price_above_weekly_cloud and price_above_daily_cloud and 
                tenkan_above_kijun and price_above_tenkan):
                signals[i] = 0.25
                position = 1
            # Short: price below daily cloud, Tenkan < Kijun, price closes below Tenkan
            elif (price_below_weekly_cloud and close[i] < daily_cloud_bottom and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < tenkan_sen_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Tenkan or below daily cloud
            if close[i] < tenkan_sen_aligned[i] or close[i] < daily_cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Tenkan or above daily cloud
            if close[i] > tenkan_sen_aligned[i] or close[i] > daily_cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals