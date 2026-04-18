#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Volume Confirmation
Hypothesis: Ichimoku Tenkan/Kijun cross above/below cloud with volume confirmation captures trend continuation in both bull/bear markets. Cloud acts as dynamic support/resistance, reducing false signals. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                      pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun_period)
    
    # Volume confirmation: volume > 1.3x 26-period EMA
    vol_ema = pd.Series(volume).ewm(span=kijun_period, adjust=False, min_periods=kijun_period).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = kijun_period + senkou_span_b_period  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        span_a = senkou_span_a[i]
        span_b = senkou_span_b[i]
        chikou = chikou_span[i] if not np.isnan(chikou_span[i]) else 0
        vol_conf = vol_ratio[i] > 1.3
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Bullish: Tenkan > Kijun, price above cloud, volume confirmation
            if tenkan > kijun and price > cloud_top and vol_conf:
                signals[i] = 0.25
                position = 1
            # Bearish: Tenkan < Kijun, price below cloud, volume confirmation
            elif tenkan < kijun and price < cloud_bottom and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Tenkan < Kijun or price below cloud
            if tenkan < kijun or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Tenkan > Kijun or price above cloud
            if tenkan > kijun or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Volume"
timeframe = "6h"
leverage = 1.0