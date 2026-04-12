#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_ichimoku_trend_v1
# Uses Ichimoku cloud (Tenkan/Kijun/Senkou) from 1d and 1w charts to filter 6s trend direction.
# Long when price > cloud and Tenkan > Kijun (bullish), short when price < cloud and Tenkan < Kijun (bearish).
# Uses weekly timeframe for stronger trend filter to avoid whipsaws in ranging markets.
# Target: 15-30 trades/year per symbol for low friction and high win rate.
name = "6h_1d_1w_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d and 1w data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 52 or len(df_1w) < 52:  # need enough for 26-period calculations
        return np.zeros(n)
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Ichimoku for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_1d = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                 pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Calculate Ichimoku for 1w (same parameters)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w = (pd.Series(high_1w).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                 pd.Series(low_1w).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    kijun_1w = (pd.Series(high_1w).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                pd.Series(low_1w).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    senkou_span_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    senkou_span_b_1w = (pd.Series(high_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(low_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    # Tenkan and Kijun are plotted with the current period (no forward shift)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w.values)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w.values)
    
    # Senkou spans are plotted 26 periods ahead, so we need to shift back 26 periods for alignment
    # This is handled by align_htf_to_ltf which uses the close time of the HTF bar
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w.values)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w.values)
    
    # Calculate cloud boundaries (top and bottom of cloud)
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top_1d = np.maximum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_top_1w = np.maximum(senkou_span_a_1w_aligned, senkou_span_b_1w_aligned)
    cloud_bottom_1w = np.minimum(senkou_span_a_1w_aligned, senkou_span_b_1w_aligned)
    
    # Use both 1d and 1w cloud for stronger filter - price must be outside both clouds
    cloud_top = np.maximum(cloud_top_1d, cloud_top_1w)
    cloud_bottom = np.minimum(cloud_bottom_1d, cloud_bottom_1w)
    
    # Trend condition: Tenkan > Kijun on both timeframes for bullish, vice versa for bearish
    bullish_tenkan_kijun = (tenkan_1d_aligned > kijun_1d_aligned) & (tenkan_1w_aligned > kijun_1w_aligned)
    bearish_tenkan_kijun = (tenkan_1d_aligned < kijun_1d_aligned) & (tenkan_1w_aligned < kijun_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after warmup for Ichimoku calculations
        # Skip if any values are not ready
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price above cloud AND Tenkan > Kijun on both timeframes
        if close[i] > cloud_top[i] and bullish_tenkan_kijun[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price below cloud AND Tenkan < Kijun on both timeframes
        elif close[i] < cloud_bottom[i] and bearish_tenkan_kijun[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite cloud penetration
        elif close[i] < cloud_bottom[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > cloud_top[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals