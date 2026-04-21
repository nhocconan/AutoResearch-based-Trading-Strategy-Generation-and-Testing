#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_Trend_V1
Hypothesis: 6h Ichimoku system with TK cross and cloud filter from 1d HTF. Uses weekly trend filter (price above/below weekly cloud) to avoid counter-trend trades. TK cross (Tenkan-Kijun) provides timely entries while cloud acts as dynamic support/resistance. Weekly trend filter ensures we only trade in the direction of the higher timeframe momentum. Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag. Works in bull markets via trend continuation and in bear markets via trend reversals aligned with weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku, 1w for trend filter)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d Ichimoku components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === 1w Ichimoku cloud for trend filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Tenkan and Kijun
    highest_tenkan_1w = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan_1w = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1w = (highest_tenkan_1w + lowest_tenkan_1w) / 2
    
    highest_kijun_1w = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun_1w = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1w = (highest_kijun_1w + lowest_kijun_1w) / 2
    
    # Weekly Senkou Span A and B
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    highest_senkou_b_1w = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b_1w = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1w = (highest_senkou_b_1w + lowest_senkou_b_1w) / 2
    
    # Align weekly Ichimoku components
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) 
            or np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])
            or np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i])
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Determine cloud boundaries (senkou_a and senkou_b)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine weekly cloud boundaries for trend filter
        weekly_upper_cloud = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        weekly_lower_cloud = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        # TK cross signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        if position == 0:
            # Long: TK cross up + price above cloud + weekly uptrend + volume
            if (tk_cross_up and price > upper_cloud and 
                price > weekly_upper_cloud and vol_ok):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TK cross down + price below cloud + weekly downtrend + volume
            elif (tk_cross_down and price < lower_cloud and 
                  price < weekly_lower_cloud and vol_ok):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: TK cross down OR price falls below cloud
            if tk_cross_down or price < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TK cross up OR price rises above cloud
            if tk_cross_up or price > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_Trend_V1"
timeframe = "6h"
leverage = 1.0