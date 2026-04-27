#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: Use Ichimoku Tenkan/Kijun cross as entry signal with 1d cloud as trend filter on 6h timeframe.
The Ichimoku cloud provides dynamic support/resistance and trend direction. TK cross indicates momentum shifts.
Combined with weekly volatility filter to avoid choppy periods. Designed for 50-150 trades over 4 years.
Works in bull via bullish TK cross above cloud, bear via bearish TK cross below cloud.
"""

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
    
    # Calculate Ichimoku components on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Weekly volatility filter: avoid low volatility periods
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly ATR(5) normalized by price
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    atr_norm = atr_5 / close_1w
    atr_norm_aligned = align_htf_to_ltf(prices, df_1w, atr_norm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Ichimoku calculations
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(atr_norm_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        vol_filter = atr_norm_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Volatility filter: require sufficient volatility (above 20th percentile)
        # Using fixed threshold as proxy - weekly ATR% > 0.01
        vol_ok = vol_filter > 0.01
        
        if position == 0:
            # Long: TK bullish cross AND price above cloud AND sufficient volatility
            if tenkan_val > kijun_val and close[i] > cloud_top and vol_ok:
                signals[i] = size
                position = 1
            # Short: TK bearish cross AND price below cloud AND sufficient volatility
            elif tenkan_val < kijun_val and close[i] < cloud_bottom and vol_ok:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TK bearish cross OR price drops below cloud bottom
            if tenkan_val < kijun_val or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK bullish cross OR price rises above cloud top
            if tenkan_val > kijun_val or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0