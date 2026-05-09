#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_IchimokuCloud_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:  # Need at least 26 days for Senkou B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 1d trend: EMA25
    ema25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema25_1d)
    
    # Volume filter: current 6h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 26)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema25_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        ema25 = ema25_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Enter long: TK cross bullish + price above cloud + above EMA25 + volume filter
            if (tenkan > kijun and 
                close[i] > cloud_top and 
                close[i] > ema25 and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross bearish + price below cloud + below EMA25 + volume filter
            elif (tenkan < kijun and 
                  close[i] < cloud_bottom and 
                  close[i] < ema25 and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish or price below cloud
            if tenkan < kijun or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish or price above cloud
            if tenkan > kijun or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals