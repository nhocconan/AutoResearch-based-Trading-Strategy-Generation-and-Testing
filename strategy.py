# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need enough data for Ichimoku (52-period lookback)
    
    for i in range(start_idx, n):
        # Skip if daily trend or Ichimoku data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long conditions: TK cross bullish + price above cloud + daily uptrend
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and  # TK cross bullish
                close[i] > cloud_top and  # price above cloud
                close[i] > ema34_1d_aligned[i]):  # daily uptrend
                signals[i] = 0.25
                position = 1
            # Short conditions: TK cross bearish + price below cloud + daily downtrend
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and  # TK cross bearish
                  close[i] < cloud_bottom and  # price below cloud
                  close[i] < ema34_1d_aligned[i]):  # daily downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when TK cross bearish or price falls below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or  # TK cross bearish
                close[i] < cloud_top):  # price below cloud top
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when TK cross bullish or price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or  # TK cross bullish
                close[i] > cloud_bottom):  # price above cloud bottom
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals