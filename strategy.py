#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend_v1
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun (TK) cross with cloud filter from 1d timeframe provides high-probability entries aligned with the daily trend. The 1d Ichimoku cloud acts as a strong trend filter (price above/below cloud), while the 6h TK cross provides precise timing. This combination reduces false signals and works in both bull and bear markets by only taking trades in the direction of the higher timeframe trend. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1d data ONCE before loop for HTF Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (max_high_52 + min_low_52) / 2
    
    # Align 1d Ichimoku components to 6h timeframe (with proper delay for completed cloud)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d, additional_delay_bars=26)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d, additional_delay_bars=26)
    
    # Calculate 6h Ichimoku components for TK cross
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_9_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_9_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (max_high_9_6h + min_low_9_6h) / 2
    
    max_high_26_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_26_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (max_high_26_6h + min_low_26_6h) / 2
    
    # TK cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Cloud filter: price above/both Senkou lines (bullish) or below/both (bearish)
    # Note: In Ichimoku, the cloud is bullish when Senkou A > Senkou B
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun)
    start_idx = max(52, 26)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or
            np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long: TK cross up + price above 1d cloud
        if tk_cross_up[i] and price_above_cloud[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short: TK cross down + price below 1d cloud
        elif tk_cross_down[i] and price_below_cloud[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit: TK cross in opposite direction (let winners run but reverse on signal)
        elif position == 1 and tk_cross_down[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and tk_cross_up[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0