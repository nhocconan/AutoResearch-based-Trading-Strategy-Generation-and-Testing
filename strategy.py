#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day timeframe filter
# Long when Tenkan-sen > Kijun-sen AND price > Kumo cloud AND Tenkan/Kijun > Kumo (bullish alignment)
# Short when Tenkan-sen < Kijun-sen AND price < Kumo cloud AND Tenkan/Kijun < Kumo (bearish alignment)
# Exit when Tenkan-sen crosses back below/above Kijun-sen
# Uses Ichimoku for multi-line trend confirmation with daily timeframe filter to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for entry)
    
    # Calculate 1d trend filter: price vs 200 EMA
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (52 for Senkou B + 26 shift buffer)
    start = 80
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Kumo cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = min(senkou_span_a[i], senkou_span_b[i])
        
        if position == 0:
            # Long setup: bullish TK cross + price above cloud + above 1d EMA200
            if (tenkan_sen[i] > kijun_sen[i] and 
                price > upper_cloud and 
                price > ema200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: bearish TK cross + price below cloud + below 1d EMA200
            elif (tenkan_sen[i] < kijun_sen[i] and 
                  price < lower_cloud and 
                  price < ema200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Tenkan-sen crosses below Kijun-sen
            if tenkan_sen[i] < kijun_sen[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Tenkan-sen crosses above Kijun-sen
            if tenkan_sen[i] > kijun_sen[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_1dEMA200_Filter"
timeframe = "6h"
leverage = 1.0