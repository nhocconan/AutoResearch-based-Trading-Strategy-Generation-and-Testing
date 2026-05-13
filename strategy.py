#!/usr/bin/env python3
name = "1d_Ichimoku_Cloud_Turn_Signal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52)
    conversion_period = 9
    base_period = 26
    lagging_span_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    max_high_9 = pd.Series(high).rolling(window=conversion_period, min_periods=conversion_period).max().values
    min_low_9 = pd.Series(low).rolling(window=conversion_period, min_periods=conversion_period).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    max_high_26 = pd.Series(high).rolling(window=base_period, min_periods=base_period).max().values
    min_low_26 = pd.Series(low).rolling(window=base_period, min_periods=base_period).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    max_high_52 = pd.Series(high).rolling(window=lagging_span_period, min_periods=lagging_span_period).max().values
    min_low_52 = pd.Series(low).rolling(window=lagging_span_period, min_periods=lagging_span_period).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Chikou Span (Lagging Span): Close shifted back 26 periods
    chikou_span = np.roll(close, base_period)  # shifted right by 26
    chikou_span[:base_period] = np.nan  # first 26 values invalid
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # need Senkou B values
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(chikou_span[i]) or np.isnan(sma50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # LONG: Price above cloud, Tenkan > Kijun, Chikou above price 26 periods ago
            if (close[i] > cloud_top and 
                tenkan_sen[i] > kijun_sen[i] and 
                chikou_span[i] > close[i - base_period]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud, Tenkan < Kijun, Chikou below price 26 periods ago
            elif (close[i] < cloud_bottom and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  chikou_span[i] < close[i - base_period]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below cloud or Tenkan < Kijun
            if close[i] < cloud_top or tenkan_sen[i] < kijun_sen[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above cloud or Tenkan > Kijun
            if close[i] > cloud_bottom or tenkan_sen[i] > kijun_sen[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals