#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Kumo (cloud) AND weekly Tenkan/Kijun cross bullish AND volume > 1.5x 20-period average.
# Short when price breaks below Kumo AND weekly Tenkan/Kijun cross bearish AND volume > 1.5x 20-period average.
# Exit when price re-enters the Kumo.
# Ichimoku provides dynamic support/resistance with trend confirmation from higher timeframe.
# Works in bull markets via cloud breakouts and in bear markets via breakdowns with weekly trend alignment.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "6h_IchimokuCloud_1wTrend_Volume"
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
    
    # Ichimoku Cloud (9, 26, 52)
    tenkan_len = 9
    kijun_len = 26
    senkou_span_b_len = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_len, min_periods=tenkan_len).max() + 
                  pd.Series(low).rolling(window=tenkan_len, min_periods=tenkan_len).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_len, min_periods=kijun_len).max() + 
                 pd.Series(low).rolling(window=kijun_len, min_periods=kijun_len).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_len, min_periods=senkou_span_b_len).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_len, min_periods=senkou_span_b_len).min()) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For plotting, these would be shifted forward, but for breakout detection we use current values
    # Upper cloud boundary: max(Senkou Span A, Senkou Span B)
    # Lower cloud boundary: min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)
    
    # 1w Trend Filter: Weekly Tenkan/Kijun cross
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < kijun_len:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Tenkan-sen (9-period)
    tenkan_1w = (pd.Series(high_1w).rolling(window=tenkan_len, min_periods=tenkan_len).max() + 
                 pd.Series(low_1w).rolling(window=tenkan_len, min_periods=tenkan_len).min()) / 2
    
    # Weekly Kijun-sen (26-period)
    kijun_1w = (pd.Series(high_1w).rolling(window=kijun_len, min_periods=kijun_len).max() + 
                pd.Series(low_1w).rolling(window=kijun_len, min_periods=kijun_len).min()) / 2
    
    # Weekly Tenkan/Kijun cross: 1 if bullish (Tenkan > Kijun), -1 if bearish (Tenkan < Kijun)
    tk_cross_1w = np.where(tenkan_1w > kijun_1w, 1, np.where(tenkan_1w < kijun_1w, -1, 0))
    tk_cross_1w_aligned = align_htf_to_ltf(prices, df_1w, tk_cross_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_len, kijun_len, senkou_span_b_len, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or 
            np.isnan(senkou_span_b[i]) or np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(tk_cross_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper cloud, weekly TK cross bullish, volume filter
            long_cond = (close[i] > upper_cloud[i]) and (tk_cross_1w_aligned[i] > 0) and volume_filter[i]
            # Short conditions: price breaks below lower cloud, weekly TK cross bearish, volume filter
            short_cond = (close[i] < lower_cloud[i]) and (tk_cross_1w_aligned[i] < 0) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters the cloud (below upper cloud)
            if close[i] < upper_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters the cloud (above lower cloud)
            if close[i] > lower_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals