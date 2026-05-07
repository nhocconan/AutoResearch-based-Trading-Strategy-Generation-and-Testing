#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_Volume"
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
    
    # 1d Ichimoku cloud (Senkou Span A/B) and TK cross (Tenkan/Kijun)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Align to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # TK cross signals
    tk_cross_up = tenkan_6h > kijun_6h
    tk_cross_down = tenkan_6h < kijun_6h
    
    # Cloud: price above cloud (bullish) or below cloud (bearish)
    # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume confirmation: 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Wait for Senkou B and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross up + price above cloud + volume surge
            if tk_cross_up[i] and price_above_cloud[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + volume surge
            elif tk_cross_down[i] and price_below_cloud[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross down or price below cloud
            if tk_cross_down[i] or not price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross up or price above cloud
            if tk_cross_up[i] or not price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku TK cross with cloud filter on 1d timeframe captures trend changes with institutional relevance.
# Long when Tenkan crosses above Kijun (bullish momentum) AND price is above the cloud (bullish trend) with volume confirmation.
# Short when Tenkan crosses below Kijun (bearish momentum) AND price is below the cloud (bearish trend) with volume confirmation.
# Uses 6h timeframe for execution, targeting 50-150 total trades over 4 years (12-37/year).
# Cloud filter ensures we only trade in strong trends, reducing whipsaws in sideways markets.
# Works in bull markets (TK cross up + price above cloud) and bear markets (TK cross down + price below cloud).