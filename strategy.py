#!/usr/bin/env python3
# 6H_Ichimoku_TK_Cross_CloudFilter_1DTrend
# Hypothesis: Ichimoku system on 6h with trend filter from daily Ichimoku cloud (Senkou Span A/B).
# Long when TK cross > 0 AND price above daily cloud (bullish regime).
# Short when TK cross < 0 AND price below daily cloud (bearish regime).
# Uses cloud as dynamic support/resistance and trend filter, reducing whipsaws in sideways markets.
# Targets 20-40 trades/year per symbol by requiring both TK cross and cloud alignment.

name = "6H_Ichimoku_TK_Cross_CloudFilter_1DTrend"
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
    
    # Get daily data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period_kijun)  # shifted 26 periods ahead
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(period_kijun)
    
    # Calculate TK cross on 6h data
    period_tenkan_6h = 9
    period_kijun_6h = 26
    tenkan_sen_6h = (pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max() + 
                     pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min()) / 2
    kijun_sen_6h = (pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max() + 
                    pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min()) / 2
    tk_cross = tenkan_sen_6h - kijun_sen_6h  # Positive when bullish, negative when bearish
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate cloud boundaries (A and B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # Ensure we have Ichimoku data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(tk_cross[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross bullish AND price above daily cloud
            if tk_cross[i] > 0 and close[i] > cloud_top[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below daily cloud
            elif tk_cross[i] < 0 and close[i] < cloud_bottom[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross turns bearish OR price falls below cloud bottom
            if tk_cross[i] < 0 or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross turns bullish OR price rises above cloud top
            if tk_cross[i] > 0 or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals