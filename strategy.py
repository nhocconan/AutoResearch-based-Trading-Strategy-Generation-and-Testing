#!/usr/bin/env python3
"""
131859: 6h_1dICHIMOKU_TK_Cross_Cloud_Filter
Hypothesis: Ichimoku's Tenkan/Kijun cross on daily timeframe provides strong directional bias,
while the cloud acts as dynamic support/resistance. Combined with 60-period EMA trend filter
and volume confirmation on 6h, this should work in both bull/bear markets by following the
daily Ichimoku trend and avoiding counter-trend trades. Target: 50-150 total trades.
"""

name = "6h_1dICHIMOKU_TK_Cross_Cloud_Filter"
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
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() +
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() +
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # 60-period EMA for additional trend filter on 6h
    ema_60_6h = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume filter: 20-period MA on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Moderate volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(ema_60_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, above EMA, volume spike
            if (tenkan_6h[i] > kijun_6h[i] and 
                close[i] > cloud_top and 
                close[i] > ema_60_6h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, below EMA, volume spike
            elif (tenkan_6h[i] < kijun_6h[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema_60_6h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if (tenkan_6h[i] < kijun_6h[i] or close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if (tenkan_6h[i] > kijun_6h[i] or close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals