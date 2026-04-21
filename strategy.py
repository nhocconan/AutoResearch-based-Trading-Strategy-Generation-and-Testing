#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_v1
Hypothesis: 6h Ichimoku TK cross filtered by 1d cloud color (trend) and volume spike.
Long when TK cross bullish + price above 1d cloud (bullish trend) + volume confirmation.
Short when TK cross bearish + price below 1d cloud (bearish trend) + volume confirmation.
Exit on opposite TK cross or cloud color change. Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via 1d cloud trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud and trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Ichimoku cloud calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # === 6h Ichimoku for TK cross ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tenkan_sen_6h = (pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                     pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun_sen_6h = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                    pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # TK cross signals
    tk_cross_bullish = tenkan_sen_6h > kijun_sen_6h
    tk_cross_bearish = tenkan_sen_6h < kijun_sen_6h
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) 
            or np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])
            or np.isnan(tk_cross_bullish[i]) or np.isnan(tk_cross_bearish[i])
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        volume_now = volume[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        tk_bull = tk_cross_bullish[i]
        tk_bear = tk_cross_bearish[i]
        vol_avg = vol_ma[i]
        
        # Cloud color: green (bullish) when span_a > span_b, red (bearish) when span_a < span_b
        cloud_bullish = span_a > span_b
        cloud_bearish = span_a < span_b
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Enter long: bullish TK cross + price above cloud + bullish cloud + volume
            long_condition = tk_bull and (price > max(span_a, span_b)) and cloud_bullish and volume_confirmed
            # Enter short: bearish TK cross + price below cloud + bearish cloud + volume
            short_condition = tk_bear and (price < min(span_a, span_b)) and cloud_bearish and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish TK cross OR price below cloud OR cloud turns bearish
            if tk_bear or (price < min(span_a, span_b)) or cloud_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross OR price above cloud OR cloud turns bullish
            if tk_bull or (price > max(span_a, span_b)) or cloud_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_v1"
timeframe = "6h"
leverage = 1.0