#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud (Tenkan/Kijun/Senkou Span) with 1d trend filter
# Uses Kumo (cloud) twist and TK cross for entry signals, filtered by 1d EMA200 trend.
# Works in bull markets (buy above cloud) and bear markets (sell below cloud).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_6h).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_6h).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_6h).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_6h).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_6h).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Calculate EMA200 on 1d for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b.values)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema200_1d_aligned[i])):
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Long entry: TK cross bullish + price above cloud + price above 1d EMA200
        if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
            tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and  # TK cross just happened
            close[i] > cloud_top and
            close[i] > ema200_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: TK cross bearish + price below cloud + price below 1d EMA200
        elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
              tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and  # TK cross just happened
              close[i] < cloud_bottom and
              close[i] < ema200_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse TK cross or price enters cloud
        elif position == 1 and (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                                close[i] < cloud_top):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                                 close[i] > cloud_bottom):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross"
timeframe = "6h"
leverage = 1.0