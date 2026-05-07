# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Trend_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): close shifted back 26 periods
    chikou_span = pd.Series(close_1d).shift(26)
    
    # Cloud top and bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span.values)
    
    # Volume spike detection: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 24)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, Chikou above price 26 periods ago, volume
            tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            price_above_cloud = close[i] > cloud_top_aligned[i]
            chikou_above = chikou_span_aligned[i] > close[i - 26] if i >= 26 else False
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            
            if tk_bullish and price_above_cloud and chikou_above and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, Chikou below price 26 periods ago, volume
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < cloud_bottom_aligned[i] and 
                  chikou_span_aligned[i] < close[i - 26] if i >= 26 else False and
                  vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bearish OR price drops below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                close[i] < cloud_top_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish OR price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                close[i] > cloud_bottom_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku cloud system with TK cross and volume confirmation on 6h timeframe
# - Tenkan/Kijun cross provides momentum signals
# - Price relative to cloud acts as trend filter (bullish above, bearish below)
# - Chikou span confirms momentum (must be above/below price 26 periods ago)
# - Volume spike (1.5x average) ensures institutional participation
# - Works in bull markets (buy signals above cloud) and bear (sell signals below cloud)
# - Cloud acts as dynamic support/resistance, reducing whipsaws
# - Position size 0.25 targets 30-60 trades/year, avoiding excessive fees
# - Ichimoku is a complete system that works across market regimes when properly filtered