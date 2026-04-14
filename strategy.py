#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    for i in range(8, len(high_1d)):
        period9_high[i] = np.max(high_1d[i-8:i+1])
        period9_low[i] = np.min(low_1d[i-8:i+1])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    for i in range(25, len(high_1d)):
        period26_high[i] = np.max(high_1d[i-25:i+1])
        period26_low[i] = np.min(low_1d[i-25:i+1])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    for i in range(51, len(high_1d)):
        period52_high[i] = np.max(high_1d[i-51:i+1])
        period52_low[i] = np.min(low_1d[i-51:i+1])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate volume spike indicator on 6h data
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(52, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_6h[i]) or 
            np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or 
            np.isnan(senkou_span_b_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = min(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        if position == 0:
            # Long: TK cross above cloud with volume spike
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] and  # TK cross bullish
                close[i] > cloud_top and                 # Price above cloud
                volume_ratio > 2.0):                     # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: TK cross below cloud with volume spike
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and  # TK cross bearish
                  close[i] < cloud_bottom and              # Price below cloud
                  volume_ratio > 2.0):                     # Volume confirmation
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: TK cross below cloud or price below cloud
            if (tenkan_sen_6h[i] < kijun_sen_6h[i] or 
                close[i] < cloud_bottom):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: TK cross above cloud or price above cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] or 
                close[i] > cloud_top):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Ichimoku_TK_Cross_Cloud_Volume"
timeframe = "6h"
leverage = 1.0