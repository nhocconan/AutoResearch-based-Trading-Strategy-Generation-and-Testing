#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: Use daily Ichimoku cloud as trend filter and 6h Tenkan/Kijun cross for entries.
Long when Tenkan > Kijun and price above daily cloud; short when Tenkan < Kijun and price below daily cloud.
Ichimoku provides dynamic support/resistance and trend direction, reducing false signals.
Designed for low trade frequency (12-37/year) to minimize fee drag while capturing trends.
"""

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
    
    # Calculate Tenkan-sen (9-period) and Kijun-sen (26-period) on 6h
    period_tenkan = 9
    period_kijun = 26
    
    # Tenkan-sen: (highest high + lowest low) / 2 over 9 periods
    high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen: (highest high + lowest low) / 2 over 26 periods
    high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_26 + low_26) / 2
    
    # Daily Ichimoku cloud (Senkou Span A and B)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (9) and Kijun-sen (26) on daily
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B: (52-period high + low) / 2, shifted 26 periods ahead
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52_1d + low_52_1d) / 2)
    
    # Align daily components to 6s timeframe (wait for daily close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Kumo (cloud) top and bottom
    kumotop = np.where(senkou_a_aligned >= senkou_b_aligned, senkou_a_aligned, senkou_b_aligned)
    kumobottom = np.where(senkou_a_aligned <= senkou_b_aligned, senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 26 + 26)  # Need 26 for Kijun and 26 for Senkou shift
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(kumotop[i]) or np.isnan(kumobottom[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        kumotop_val = kumotop[i]
        kumobottom_val = kumobottom[i]
        
        if position == 0:
            # Long: Tenkan > Kijun and price above cloud
            if tenkan_val > kijun_val and price > kumotop_val:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun and price below cloud
            elif tenkan_val < kijun_val and price < kumobottom_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Tenkan < Kijun or price below cloud
            if tenkan_val < kijun_val or price < kumobottom_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Tenkan > Kijun or price above cloud
            if tenkan_val > kijun_val or price > kumotop_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0