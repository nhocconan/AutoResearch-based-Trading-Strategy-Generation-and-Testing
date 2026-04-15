#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with Daily Filter
# Uses Ichimoku components (Tenkan, Kijun, Senkou A/B, Chikou) on 6h timeframe.
# Long when price > cloud and Tenkan > Kijun (bullish), short when price < cloud and Tenkan < Kijun (bearish).
# Filters trades using daily trend: only take longs when 1d EMA50 > EMA200, shorts when EMA50 < EMA200.
# Works in bull markets (captures uptrends via cloud) and bear markets (shorts below cloud).
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Load 1d data for daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 and EMA200
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align Ichimoku components to 6h timeframe (no shift - Ichimoku is calculated on close)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b.values)
    
    # Align daily EMAs to 6h timeframe (need to wait for daily close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Check daily trend
        daily_uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        daily_downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Long entry: price above cloud, Tenkan > Kijun, and daily uptrend
        if (close[i] > cloud_top and
            tenkan_aligned[i] > kijun_aligned[i] and
            daily_uptrend and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below cloud, Tenkan < Kijun, and daily downtrend
        elif (close[i] < cloud_bottom and
              tenkan_aligned[i] < kijun_aligned[i] and
              daily_downtrend and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Ichimoku signal or daily trend reversal
        elif position == 1 and (close[i] < cloud_bottom or tenkan_aligned[i] < kijun_aligned[i] or not daily_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > cloud_top or tenkan_aligned[i] > kijun_aligned[i] or not daily_downtrend):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_DailyTrend_Filter"
timeframe = "6h"
leverage = 1.0