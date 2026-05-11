# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1w"
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (using previous weekly bar's values)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        if position == 0:
            # Long: TK cross bullish AND price above cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                close[i] > cloud_top and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                  close[i] < cloud_bottom and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross bearish OR price below cloud
            if (tenkan_sen_6h[i] < kijun_sen_6h[i] or 
                close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: TK cross bullish OR price above cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] or 
                close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals