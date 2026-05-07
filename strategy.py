# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_1d_Ichimoku_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily timeframe
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan_sen = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun_sen = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_span_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = df_1d['close'].shift(26).values
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # Cloud top and bottom
    cloud_top = np.maximum(span_a_aligned, span_b_aligned)
    cloud_bottom = np.minimum(span_a_aligned, span_b_aligned)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 4)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(chikou_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross above, price above cloud, Chikou above price 26 periods ago, volume spike
            tk_cross = tenkan_aligned[i] > kijun_aligned[i]
            price_above_cloud = close[i] > cloud_top[i]
            chikou_above = chikou_aligned[i] > close[i]  # Chikou above current price (shifted back)
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            
            if tk_cross and price_above_cloud and chikou_above and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below, price below cloud, Chikou below price 26 periods ago, volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < cloud_bottom[i] and 
                  chikou_aligned[i] < close[i] and 
                  vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price drops below cloud base or TK cross turns bearish
            if close[i] < cloud_bottom[i] or tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above cloud top or TK cross turns bullish
            if close[i] > cloud_top[i] or tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku with cloud filter and TK cross
# - Uses Ichimoku cloud from daily timeframe as dynamic support/resistance
# - Tenkan/Kijun cross provides momentum signal
# - Chikou span confirms trend strength (price vs lagged price)
# - Cloud acts as trend filter: only long when price above cloud, short when below
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull and bear markets via cloud position
# - Position size 0.25 targets ~20-60 trades/year, avoiding fee drag
# - Ichimoku is proven effective in crypto markets for trend identification
# - Daily Ichimoku provides stable signals for 6x execution timeframe