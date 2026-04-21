#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun cross filtered by 1d cloud (Senkou Span A/B) for trend alignment.
Enter long when TK crosses above AND price > cloud (bullish regime).
Enter short when TK crosses below AND price < cloud (bearish regime).
Exit on opposite TK cross or when price re-enters cloud (trend exhaustion).
Uses discrete sizing (0.0, ±0.25) to limit fee churn. Designed for 60-120 trades/year.
Works in bull/bear via 1d cloud filter that adapts to regime (cloud acts as dynamic support/resistance).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # === 6h Ichimoku components (Tenkan, Kijun) ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # === 1d Ichimoku cloud (Senkou Span A/B) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Senkou Span A = (Tenkan + Kijun)/2 plotted 26 periods ahead
    # But for filtering, we use current cloud (already shifted in calculation)
    period_tenkan_1d = 9
    period_kijun_1d = 26
    high_tenkan_1d = pd.Series(high_1d).rolling(window=period_tenkan_1d, min_periods=period_tenkan_1d).max().values
    low_tenkan_1d = pd.Series(low_1d).rolling(window=period_tenkan_1d, min_periods=period_tenkan_1d).min().values
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    high_kijun_1d = pd.Series(high_1d).rolling(window=period_kijun_1d, min_periods=period_kijun_1d).max().values
    low_kijun_1d = pd.Series(low_1d).rolling(window=period_kijun_1d, min_periods=period_kijun_1d).min().values
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    senkou_span_a = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B = (52-period high + low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2
    
    # The cloud is between Senkou Span A and B
    # For trend filter: price > max(span A, B) = bullish, price < min(span A, B) = bearish
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align 1d cloud to 6h (wait for completed 1d bar)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # need 52 for Senkou Span B
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) 
            or np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # TK cross: Tenkan crosses Kijun
            tk_cross_up = tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]
            tk_cross_down = tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]
            
            # Cloud filter: price above cloud (bullish) or below cloud (bearish)
            price_above_cloud = price > cloud_top_aligned[i]
            price_below_cloud = price < cloud_bottom_aligned[i]
            
            # Long: TK cross up + bullish cloud (price above cloud)
            if tk_cross_up and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + bearish cloud (price below cloud)
            elif tk_cross_down and price_below_cloud:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: TK cross down OR price re-enters cloud (trend weakness)
            tk_cross_down = tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]
            price_in_cloud = price >= cloud_bottom_aligned[i] and price <= cloud_top_aligned[i]
            
            if tk_cross_down or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: TK cross up OR price re-enters cloud (trend weakness)
            tk_cross_up = tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]
            price_in_cloud = price >= cloud_bottom_aligned[i] and price <= cloud_top_aligned[i]
            
            if tk_cross_up or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter_v1"
timeframe = "6h"
leverage = 1.0