#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Ichimoku cloud with daily Tenkan/Kijun cross and volume confirmation
# Long when price > weekly cloud AND Tenkan > Kijun (daily) with volume > 1.5x average
# Short when price < weekly cloud AND Tenkan < Kijun (daily) with volume > 1.5x average
# Uses weekly Ichimoku for major trend structure, daily TK cross for momentum, volume for confirmation
# Target: 15-30 trades per year (60-120 over 4 years) with 0.25 position sizing

name = "6h_1wIchimoku_1dTKCross_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Ichimoku components
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 1 year of weekly data
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(df_1w['close']).shift(26).values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1w, senkou_b)
    chikou_6h = align_htf_to_ltf(prices, df_1w, chikou)
    
    # Calculate daily Tenkan/Kijun cross for momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Daily Tenkan-sen
    high_9d = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9d = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9d + low_9d) / 2
    
    # Daily Kijun-sen
    high_26d = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26d = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26d + low_26d) / 2
    
    # Align daily TK cross to 6h timeframe
    tenkan_1d_6h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_6h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after weekly Ichimoku warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(tenkan_1d_6h[i]) or np.isnan(kijun_1d_6h[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long: price above cloud, bullish TK cross (daily), volume confirmation
            if (close[i] > cloud_top and 
                tenkan_1d_6h[i] > kijun_1d_6h[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, bearish TK cross (daily), volume confirmation
            elif (close[i] < cloud_bottom and 
                  tenkan_1d_6h[i] < kijun_1d_6h[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below cloud or bearish TK cross
            if close[i] < cloud_bottom or tenkan_1d_6h[i] < kijun_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above cloud or bullish TK cross
            if close[i] > cloud_top or tenkan_1d_6h[i] > kijun_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals