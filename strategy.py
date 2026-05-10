#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
# Hypothesis: Ichimoku Tenkan-Kijun cross on 6h chart with price relative to Kumo (cloud) from 1d chart.
# In bullish regime (price above 1d Kumo), go long on TK cross above cloud, short on cross below.
# In bearish regime (price below 1d Kumo), go short on TK cross below cloud, long on cross above.
# Uses Kumo as dynamic support/resistance to avoid whipsaws. Targets 15-30 trades/year.

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
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
    
    # 1d data for Kumo (cloud) calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components on 1d: Tenkan (9), Kijun (26), Senkou A/B (52 displaced)
    # Tenkan-sen: (9-period high + low)/2
    def highest_n(arr, n):
        res = np.full_like(arr, np.nan)
        for i in range(n-1, len(arr)):
            res[i] = np.max(arr[i-n+1:i+1])
        return res
    
    def lowest_n(arr, n):
        res = np.full_like(arr, np.nan)
        for i in range(n-1, len(arr)):
            res[i] = np.min(arr[i-n+1:i+1])
        return res
    
    tenkan_1d = (highest_n(high_1d, 9) + lowest_n(low_1d, 9)) / 2
    kijun_1d = (highest_n(high_1d, 26) + lowest_n(low_1d, 26)) / 2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    senkou_b_1d = (highest_n(high_1d, 52) + lowest_n(low_1d, 52)) / 2
    
    # Kumo (cloud) boundaries: Senkou A and B shifted forward 26 periods
    # For cloud at time t, we need Senkou values from t-26
    senkou_a_shifted = np.roll(senkou_a_1d, 26)
    senkou_b_shifted = np.roll(senkou_b_1d, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Kumo top/bottom
    kumo_top_1d = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom_1d = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Align 1d Ichimoku to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # 6h Ichimoku for TK cross
    tenkan_6h = (highest_n(high, 9) + lowest_n(low, 9)) / 2
    kijun_6h = (highest_n(high, 26) + lowest_n(low, 26)) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough for Ichimoku
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or \
           np.isnan(kumo_top_1d_aligned[i]) or np.isnan(kumo_bottom_1d_aligned[i]) or \
           np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price relative to 1d Kumo
        price_above_kumo = close[i] > kumo_top_1d_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_1d_aligned[i]
        price_in_kumo = ~(price_above_kumo | price_below_kumo)
        
        # TK cross signals
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 0:
            # Bullish regime: price above cloud -> long on TK cross up
            if price_above_kumo and tk_cross_up:
                signals[i] = 0.25
                position = 1
            # Bearish regime: price below cloud -> short on TK cross down
            elif price_below_kumo and tk_cross_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross down or price drops below cloud
            if tk_cross_down or price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up or price rises above cloud
            if tk_cross_up or price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals