#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Cloud_Filter"
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe (need 26 periods for Kijun)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 6)  # Need 52 for Senkou B and 6 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Enter long: TK cross bullish, price above cloud, volume confirmation
            if tenkan > kijun and close[i] > cloud_top and vol > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross bearish, price below cloud, volume confirmation
            elif tenkan < kijun and close[i] < cloud_bottom and vol > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if tenkan < kijun or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if tenkan > kijun or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals