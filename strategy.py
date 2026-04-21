#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTK_Cross_v4
Hypothesis: Ichimoku cloud with TK cross on 6h, filtered by 1d trend (price above/below 1d Kijun-sen) and volume confirmation. Designed for low trade frequency (~15-25/year) to minimize fee drag. Works in bull/bear markets by only taking trades in direction of 1d trend, using cloud as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Ichimoku components for trend filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    highest_10_1d = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_10_1d = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (highest_10_1d + lowest_10_1d) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    highest_26_1d = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_26_1d = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (highest_26_1d + lowest_26_1d) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    highest_52_1d = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_52_1d = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (highest_52_1d + lowest_52_1d) / 2
    
    # Align 1d Ichimoku components to 6h timeframe (no extra delay needed for these)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # === 6h Ichimoku components for entry signal ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (6h)
    highest_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    lowest_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (highest_9 + lowest_9) / 2
    
    # Kijun-sen (6h)
    highest_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    lowest_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (highest_26 + lowest_26) / 2
    
    # Senkou Span A (6h)
    senkou_a_6h = (tenkan_6h + kijun_6h) / 2
    # Senkou Span B (6h)
    highest_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    lowest_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b_6h = (highest_52 + lowest_52) / 2
    
    # === 6h volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma[np.isnan(vol_ma)] = 1.0  # avoid division by zero
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # 6h Ichimoku cloud boundaries
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        # 1d trend filter: price relative to 1d Kijun-sen
        trend_1d = kijun_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + above 1d Kijun + volume spike
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] and  # TK cross bullish
                price_close > cloud_top and price_close > trend_1d and vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + below 1d Kijun + volume spike
            elif (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] and  # TK cross bearish
                  price_close < cloud_bottom and price_close < trend_1d and vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when TK cross reverses or price exits cloud in opposite direction
            if position == 1:
                # Exit on bearish TK cross or price falls below cloud
                if (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]) or price_close < cloud_bottom:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish TK cross or price rises above cloud
                if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]) or price_close > cloud_top:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTK_Cross_v4"
timeframe = "6h"
leverage = 1.0