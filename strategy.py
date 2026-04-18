#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_v1
Hypothesis: Uses Ichimoku cloud on 6h with daily trend filter to capture trend continuation in both bull and bear markets. The cloud acts as dynamic support/resistance, and the daily trend ensures alignment with higher timeframe bias. Tenkan/Kijun cross provides entry signal, with cloud breakout confirmation. This combination reduces false signals and works in trending markets across regimes.
"""

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
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    tenkan_sen = ((high_series.rolling(window=9, min_periods=9).max() + 
                   low_series.rolling(window=9, min_periods=9).min()) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = ((high_series.rolling(window=26, min_periods=26).max() + 
                  low_series.rolling(window=26, min_periods=26).min()) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = ((high_series.rolling(window=52, min_periods=52).max() + 
                      low_series.rolling(window=52, min_periods=52).min()) / 2).values
    
    # Daily trend filter: price vs 60 EMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_60_1d = pd.Series(close_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    ema_60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_60_1d)
    
    signals = np.zeros(n)
    
    start_idx = 52  # Need Senkou B
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_60_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        span_a = senkou_span_a[i]
        span_b = senkou_span_b[i]
        ema_1d = ema_60_1d_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Bullish: price above cloud, Tenkan > Kijun, and above daily EMA
        if price > cloud_top and tenkan > kijun and price > ema_1d:
            signals[i] = 0.25
        # Bearish: price below cloud, Tenkan < Kijun, and below daily EMA
        elif price < cloud_bottom and tenkan < kijun and price < ema_1d:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0