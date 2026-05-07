#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_12hTrend
# Hypothesis: Ichimoku Tenkan/Kijun cross with cloud filter (from 1d) and 12h trend filter provides
# high-probability entries in trending markets while avoiding whipsaws in ranges.
# Works in both bull and bear by only trading in direction of 12h trend. Target: 15-25 trades/year.

name = "6h_Ichimoku_TK_Cross_CloudFilter_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku cloud and 12h data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_1d) < 52 or len(df_12h) < 26:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (9-period): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (26-period): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 12h trend filter (EMA 26)
    close_12h = df_12h['close'].values
    ema_26_12h = pd.Series(close_12h).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_12h_6h = align_htf_to_ltf(prices, df_12h, ema_26_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(ema_26_12h_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = min(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        if position == 0:
            # Long: TK cross above + price above cloud + 12h uptrend
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1] and  # fresh cross
                close[i] > cloud_top and 
                close[i] > ema_26_12h_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross below + price below cloud + 12h downtrend
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                  tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1] and  # fresh cross
                  close[i] < cloud_bottom and 
                  close[i] < ema_26_12h_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross below OR price below cloud
            if (tenkan_sen_6h[i] < kijun_sen_6h[i] or 
                close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross above OR price above cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] or 
                close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals