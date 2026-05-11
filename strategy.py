#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Kumo_Twist_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max().values + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max().values + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                     pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo (Cloud) boundaries
    kumomax = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumomin = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need enough data for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if 12h trend or Ichimoku data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(kumomax[i]) or 
            np.isnan(kumomin[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: TK cross bullish + price above cloud + 12h uptrend
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and  # TK cross bullish
                close[i] > kumomax[i] and  # price above cloud
                close[i] > ema50_12h_aligned[i]):  # 12h uptrend
                signals[i] = 0.25
                position = 1
            # Short conditions: TK cross bearish + price below cloud + 12h downtrend
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and  # TK cross bearish
                  close[i] < kumomin[i] and  # price below cloud
                  close[i] < ema50_12h_aligned[i]):  # 12h downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when TK cross bearish or price drops below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                close[i] < kumomax[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when TK cross bullish or price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                close[i] > kumomin[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals