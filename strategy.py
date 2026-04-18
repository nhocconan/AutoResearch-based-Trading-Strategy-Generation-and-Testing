#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1D_v2
Hypothesis: On 6h timeframe, use daily Ichimoku cloud for trend filter and TK cross for entry timing.
In bull markets (price above cloud), buy Tenkan-Kijun cross up; in bear markets (price below cloud), sell cross down.
Weekly trend filter avoids counter-trend trades in strong trends.
Designed to work in both bull and bear by requiring price/cloud alignment.
Target: 15-25 trades/year to minimize fee drift while capturing medium-term momentum shifts.
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
    
    # Daily Ichimoku components (conversion, base, leading spans)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Weekly trend filter: price vs weekly EMA200
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema200 = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(52, 26)  # Warmup for Ichimoku
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(weekly_ema200[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        weekly_ema = weekly_ema200[i]
        
        # Cloud boundaries: top is max(Senkou A, Senkou B), bottom is min
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Determine if price is above or below cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # TK cross signals
        tk_cross_up = tenkan_val > kijun_val and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_val < kijun_val and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 0:
            # Long: price above cloud + TK cross up + price above weekly EMA200 (bullish alignment)
            if price_above_cloud and tk_cross_up and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down + price below weekly EMA200 (bearish alignment)
            elif price_below_cloud and tk_cross_down and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TK cross down OR price re-enters cloud OR weekly trend fails
            if tk_cross_down or (price <= cloud_top and price >= cloud_bottom) or price < weekly_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TK cross up OR price re-enters cloud OR weekly trend fails
            if tk_cross_up or (price <= cloud_top and price >= cloud_bottom) or price > weekly_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1D_v2"
timeframe = "6h"
leverage = 1.0