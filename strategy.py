#!/usr/bin/env python3
"""
1d_Ichimoku_Tenkan_Kijun_Cross_1wTrend_Filter
Hypothesis: Ichimoku Tenkan-sen/Kijun-sen cross on daily chart, filtered by weekly trend (price above/below weekly Kumo cloud), generates fewer, higher-quality signals. Works in bull/bear by following weekly trend direction, reducing false signals during counter-trend moves. Target: 15-25 trades/year.
"""

name = "1d_Ichimoku_Tenkan_Kijun_Cross_1wTrend_Filter"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B: (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Kumo cloud top and bottom
    kumO_top = np.maximum(senkou_a, senkou_b)
    kumO_bottom = np.minimum(senkou_a, senkou_b)
    
    # Shift Senkou spans by 26 periods forward (for cloud)
    senkou_a_lead = np.roll(senkou_a, 26)
    senkou_b_lead = np.roll(senkou_b, 26)
    senkou_a_lead[:26] = np.nan
    senkou_b_lead[:26] = np.nan
    kumO_top_lead = np.maximum(senkou_a_lead, senkou_b_lead)
    kumO_bottom_lead = np.minimum(senkou_a_lead, senkou_b_lead)
    
    # Weekly trend filter: price relative to weekly Kumo
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Ichimoku for trend filter
    high_9_w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_w = (high_9_w + low_9_w) / 2
    
    high_26_w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_w = (high_26_w + low_26_w) / 2
    
    senkou_a_w = (tenkan_w + kijun_w) / 2
    
    high_52_w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_w = (high_52_w + low_52_w) / 2
    
    kumO_top_w = np.maximum(senkou_a_w, senkou_b_w)
    kumO_bottom_w = np.minimum(senkou_a_w, senkou_b_w)
    
    # Align weekly cloud to daily
    kumO_top_w_aligned = align_htf_to_ltf(prices, df_1w, kumO_top_w)
    kumO_bottom_w_aligned = align_htf_to_ltf(prices, df_1w, kumO_bottom_w)
    
    # Tenkan/Kijun cross signals
    tenkan_kijun_cross = tenkan - kijun
    tenkan_kijun_cross_prev = np.roll(tenkan_kijun_cross, 1)
    tenkan_kijun_cross_prev[0] = np.nan
    
    # Bullish cross: Tenkan crosses above Kijun
    bullish_cross = (tenkan_kijun_cross > 0) & (tenkan_kijun_cross_prev <= 0)
    # Bearish cross: Tenkan crosses below Kijun
    bearish_cross = (tenkan_kijun_cross < 0) & (tenkan_kijun_cross_prev >= 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        if (np.isnan(kumO_top_w_aligned[i]) or np.isnan(kumO_bottom_w_aligned[i]) or
            np.isnan(bullish_cross[i]) or np.isnan(bearish_cross[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish cross + price above weekly Kumo (bullish weekly trend)
            if bullish_cross[i] and close[i] > kumO_top_w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish cross + price below weekly Kumo (bearish weekly trend)
            elif bearish_cross[i] and close[i] < kumO_bottom_w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish cross or price drops below weekly Kumo bottom
            if bearish_cross[i] or close[i] < kumO_bottom_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish cross or price rises above weekly Kumo top
            if bullish_cross[i] or close[i] > kumO_top_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals