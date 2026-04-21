#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with Tenkan/Kijun cross and daily trend filter.
Longs when Tenkan crosses above Kijun, price above cloud, and 1d EMA50 > EMA200.
Shorts when Tenkan crosses below Kijun, price below cloud, and 1d EMA50 < EMA200.
Exit when Tenkan/Kijun cross reverses or price exits cloud.
Ichimoku provides dynamic support/resistance and trend direction, effective in both trending and ranging markets.
Designed for 15-30 trades/year to minimize fee drift while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50 and 200 period EMA for daily trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMAs to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Align Ichimoku components (no shift needed as align_htf_to_ltf handles completion)
    tenkan_sen = tenkan_sen.values
    kijun_sen = kijun_sen.values
    senkou_span_a = senkou_span_a.values
    senkou_span_b = senkou_span_b.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        tsen = tenkan_sen[i]
        ksen = kijun_sen[i]
        span_a = senkou_span_a[i]
        span_b = senkou_span_b[i]
        ema50 = ema50_1d_aligned[i]
        ema200 = ema200_1d_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Enter long: Tenkan crosses above Kijun, price above cloud, daily uptrend
            if (tsen > ksen and 
                price_close > cloud_top and 
                ema50 > ema200):
                signals[i] = 0.25
                position = 1
            # Enter short: Tenkan crosses below Kijun, price below cloud, daily downtrend
            elif (tsen < ksen and 
                  price_close < cloud_bottom and 
                  ema50 < ema200):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Tenkan/Kijun cross reverses OR price exits cloud
            exit_signal = False
            
            # Tenkan/Kijun cross reversal
            if position == 1 and tsen < ksen:
                exit_signal = True
            elif position == -1 and tsen > ksen:
                exit_signal = True
            
            # Price exits cloud
            if position == 1 and price_close < cloud_bottom:
                exit_signal = True
            elif position == -1 and price_close > cloud_top:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_TenkanKijun_Cloud_1dEMA50_200_Trend"
timeframe = "6h"
leverage = 1.0