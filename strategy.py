#!/usr/bin/env python3
# 6h_1d_Ichimoku_TF_Momentum
# Hypothesis: Ichimoku cloud on 1d provides trend direction, Tenkan/Kijun cross on 1d gives entry signal, with volume confirmation to avoid false breaks.
# Works in bull/bear: Only take longs when price above cloud and TK cross bullish, shorts when price below cloud and TK cross bearish.
# Uses 6h timeframe for execution, targeting 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_TF_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === Ichimoku components on 1d ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close_1d).shift(26)
    
    # Convert to numpy arrays
    tenkan_sen = tenkan_sen.values
    kijun_sen = kijun_sen.values
    senkou_span_a = senkou_span_a.values
    senkou_span_b = senkou_span_b.values
    chikou_span = chikou_span.values
    
    # Cloud top and bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d Ichimoku data to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Get values
        close_val = prices['close'].iloc[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        cloud_top_val = cloud_top_aligned[i]
        cloud_bottom_val = cloud_bottom_aligned[i]
        chikou_val = chikou_span_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or 
            np.isnan(cloud_top_val) or np.isnan(cloud_bottom_val) or
            np.isnan(chikou_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above cloud, TK cross bullish (Tenkan > Kijun), Chikou above price, volume confirmation
            if (close_val > cloud_top_val and  # Price above cloud
                tenkan_val > kijun_val and  # TK cross bullish
                chikou_val > close_val and  # Chikou above current price (momentum)
                vol_ratio_val > 1.5):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, TK cross bearish (Tenkan < Kijun), Chikou below price, volume confirmation
            elif (close_val < cloud_bottom_val and  # Price below cloud
                  tenkan_val < kijun_val and  # TK cross bearish
                  chikou_val < close_val and  # Chikou below current price (momentum)
                  vol_ratio_val > 1.5):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below cloud or TK cross turns bearish
            if close_val < cloud_top_val or tenkan_val < kijun_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above cloud or TK cross turns bullish
            if close_val > cloud_bottom_val or tenkan_val > kijun_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals