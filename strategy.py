#!/usr/bin/env python3
# 6H_Ichimoku_TK_Cross_CloudFilter_1d
# Hypothesis: Ichimoku TK cross (Tenkan/Kijun) on 6h with daily cloud filter provides high-probability trend entries.
# In bull markets: price above cloud + TK cross up = long. In bear markets: price below cloud + TK cross down = short.
# Cloud acts as dynamic support/resistance, reducing whipsaws. Targets 15-25 trades/year.

name = "6H_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components on 6h (Tenkan: 9, Kijun: 26, Senkou B: 52)
    period_tenkan = 9
    period_kijun = 26
    period_senkou_b = 52
    
    # Tenkan-sen: (highest high + lowest low) / 2 over period
    tenkan_high = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    tenkan_low = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen: (highest high + lowest low) / 2 over period
    kijun_high = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    kijun_low = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (kijun_high + kijun_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B: (highest high + lowest low) / 2 over period_senkou_b
    senkou_b_high = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    senkou_b_low = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (senkou_b_high + senkou_b_low) / 2
    
    # TK Cross signals
    tk_cross_up = (tenkan > kijun) & (tenkan_shift := np.roll(tenkan, 1)) <= (kijun_shift := np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (tenkan_shift := np.roll(tenkan, 1)) >= (kijun_shift := np.roll(kijun, 1))
    
    # Handle first element for roll
    tk_cross_up[0] = False
    tk_cross_down[0] = False
    
    # 1d cloud filter (Senkou Span A and B from 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < period_senkou_b:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Ichimoku components
    tenkan_1d_high = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    tenkan_1d_low = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (tenkan_1d_high + tenkan_1d_low) / 2
    
    kijun_1d_high = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    kijun_1d_low = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (kijun_1d_high + kijun_1d_low) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    senkou_b_1d_high = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    senkou_b_1d_low = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (senkou_b_1d_high + senkou_b_1d_low) / 2
    
    # Align 1d cloud to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_senkou_b, period_kijun)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross up + price above cloud
            if tk_cross_up[i] and close[i] > cloud_top[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud
            elif tk_cross_down[i] and close[i] < cloud_bottom[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross down OR price below cloud
            if tk_cross_down[i] or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up OR price above cloud
            if tk_cross_up[i] or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals