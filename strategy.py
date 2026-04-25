#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_1dKijun_Filter
Hypothesis: 6-hour Ichimoku cloud strategy with 1-day Kijun-sen trend filter.
Long when price is above cloud AND Tenkan-sen > Kijun-sen (bullish TK cross) in 1-day uptrend (close > 1d Kijun).
Short when price is below cloud AND Tenkan-sen < Kijun-sen (bearish TK cross) in 1-day downtrend (close < 1d Kijun).
Exit via opposite TK cross or when price re-enters the cloud.
Designed for ~12-37 trades/year via Ichimoku's built-in trend/filter properties and 1d trend alignment.
Works in bull via cloud support/resistance and in bear via cloud resistance/support with TK cross confirmation.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen: (highest high + lowest low)/2 over past 9 periods
    highest_tenkan = pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen: (highest high + lowest low)/2 over past 26 periods
    highest_kijun = pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A: (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B: (highest high + lowest low)/2 over past 52 periods plotted 26 periods ahead
    highest_senkou_b = pd.Series(high_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Align Ichimoku components to primary timeframe (6h)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d Kijun-sen for trend filter
    highest_kijun_1d = pd.Series(close_1d).rolling(window=26, min_periods=26).max().values
    lowest_kijun_1d = pd.Series(close_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (highest_kijun_1d + lowest_kijun_1d) / 2
    
    # Align 1d Kijun-sen to 6h timeframe
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations
    start_idx = max(100, senkou_span_b_period + 26, 26)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        kijun_1d = kijun_1d_aligned[i]
        price = close[i]
        
        # Cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a, senkou_b)
        lower_cloud = min(senkou_a, senkou_b)
        
        # TK cross signals
        bullish_tk = tenkan > kijun
        bearish_tk = tenkan < kijun
        
        if position == 0:
            # Only trade in alignment with 1d Kijun-sen trend
            if price > kijun_1d:  # 1d uptrend regime
                # Long: price above cloud AND bullish TK cross
                long_signal = (price > upper_cloud) and bullish_tk
            else:  # 1d downtrend regime
                # Short: price below cloud AND bearish TK cross
                short_signal = (price < lower_cloud) and bearish_tk
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters cloud OR bearish TK cross
            if price < upper_cloud or bearish_tk:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters cloud OR bullish TK cross
            if price > lower_cloud or bullish_tk:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_1dKijun_Filter"
timeframe = "6h"
leverage = 1.0