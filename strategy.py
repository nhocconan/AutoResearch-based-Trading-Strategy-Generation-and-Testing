#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter
# Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen and price is above Kumo cloud (from daily Ichimoku). Enter short when Tenkan-sen crosses below Kijun-sen and price is below Kumo cloud. The daily Ichimoku cloud acts as a trend filter to avoid counter-trend trades. This strategy aims to capture medium-term trends with fewer trades by requiring alignment between 6h momentum and 1d trend, reducing whipsaws in ranging markets. Targets 12-30 trades/year for low fee drag.

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter"
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
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Calculate daily Ichimoku for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Daily Tenkan-sen
    daily_tenkan = (pd.Series(daily_high).rolling(window=9, min_periods=9).max().values + 
                    pd.Series(daily_low).rolling(window=9, min_periods=9).min().values) / 2
    
    # Daily Kijun-sen
    daily_kijun = (pd.Series(daily_high).rolling(window=26, min_periods=26).max().values + 
                   pd.Series(daily_low).rolling(window=26, min_periods=26).min().values) / 2
    
    # Daily Senkou Span A
    daily_senkou_a = (daily_tenkan + daily_kijun) / 2
    
    # Daily Senkou Span B
    daily_senkou_b = (pd.Series(daily_high).rolling(window=52, min_periods=52).max().values + 
                      pd.Series(daily_low).rolling(window=52, min_periods=52).min().values) / 2
    
    # Align daily Ichimoku components to 6h timeframe
    daily_tenkan_aligned = align_htf_to_ltf(prices, df_1d, daily_tenkan)
    daily_kijun_aligned = align_htf_to_ltf(prices, df_1d, daily_kijun)
    daily_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, daily_senkou_a)
    daily_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, daily_senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable (max period 52 + buffer)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(daily_tenkan_aligned[i]) or np.isnan(daily_kijun_aligned[i]) or
            np.isnan(daily_senkou_a_aligned[i]) or np.isnan(daily_senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine if price is above or below daily Kumo cloud
        daily_kumo_top = np.maximum(daily_senkou_a_aligned[i], daily_senkou_b_aligned[i])
        daily_kumo_bottom = np.minimum(daily_senkou_a_aligned[i], daily_senkou_b_aligned[i])
        price_above_daily_kumo = close[i] > daily_kumo_top
        price_below_daily_kumo = close[i] < daily_kumo_bottom
        
        # TK cross signals
        tk_cross_above = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_below = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        if position == 0:
            # LONG: TK cross above + price above daily cloud
            if tk_cross_above and price_above_daily_kumo:
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross below + price below daily cloud
            elif tk_cross_below and price_below_daily_kumo:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross below (momentum shift)
            if tk_cross_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross above (momentum shift)
            if tk_cross_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals