#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku and trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2.0)
    
    # Align Ichimoku components to 6h timeframe (with proper shift for leading spans)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_d, senkou_span_b)
    
    # Get daily close for trend filter (price above/below 50 EMA)
    ema_50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_d_aligned = align_htf_to_ltf(prices, df_d, ema_50_d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Need enough data for Senkou B and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        ema_50 = ema_50_d_aligned[i]
        vol_filter = volume_filter[i]
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Enter long: TK cross bullish, price above cloud, above daily EMA50, volume filter
            if (tenkan > kijun and 
                close[i] > cloud_top and 
                close[i] > ema_50 and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross bearish, price below cloud, below daily EMA50, volume filter
            elif (tenkan < kijun and 
                  close[i] < cloud_bottom and 
                  close[i] < ema_50 and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish or price falls below cloud
            if (tenkan < kijun) or (close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish or price rises above cloud
            if (tenkan > kijun) or (close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals