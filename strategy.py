#!/usr/bin/env python3
# 6H_Ichimoku_Cloud_Breakout_1dTrend
# Hypothesis: Uses 1d Ichimoku for trend and 6x Tenkan/Kijun cross for entry. 
# Tenkan/Kijun cross above/below Kumo cloud triggers entry in direction of 1d trend.
# Kumo cloud acts as dynamic support/resistance, reducing whipsaw in ranging markets.
# Works in bull/bear by following 1d Ichimoku trend (Senkou Span A/B).

name = "6H_Ichimoku_Cloud_Breakout_1dTrend"
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
    volume = prices['volume'].values
    
    # 6h Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    tenkan = (high_series.rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
              low_series.rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan = tenkan.values
    
    # 6h Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun = (high_series.rolling(window=period_kijun, min_periods=period_kijun).max() + 
             low_series.rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun = kijun.values
    
    # 6h Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Will be aligned later; no shift here as align_htf_to_ltf handles timing
    
    # 6h Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b = (high_series.rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                low_series.rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_b = senkou_b.values
    
    # 1d Ichimoku for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan-sen (9-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    tenkan_1d = (high_1d_series.rolling(window=9, min_periods=9).max() + 
                 low_1d_series.rolling(window=9, min_periods=9).min()) / 2
    tenkan_1d = tenkan_1d.values
    
    # 1d Kijun-sen (26-period)
    kijun_1d = (high_1d_series.rolling(window=26, min_periods=26).max() + 
                low_1d_series.rolling(window=26, min_periods=26).min()) / 2
    kijun_1d = kijun_1d.values
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B (52-period)
    senkou_b_1d = (high_1d_series.rolling(window=52, min_periods=52).max() + 
                   low_1d_series.rolling(window=52, min_periods=52).min()) / 2
    senkou_b_1d = senkou_b_1d.values
    
    # 1d trend: price above/below Kumo (cloud)
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    trend_1d_up = close_1d > kumo_top_1d  # Bullish: price above cloud
    trend_1d_down = close_1d < kumo_bottom_1d  # Bearish: price below cloud
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Align 6h Senkou Span A and B (no additional delay needed as they are based on current/ past data)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_b)
    
    # Kumo (cloud) top and bottom for 6h
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Enough for Senkou B (52-period) + alignment
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(kumo_top[i]) or 
            np.isnan(kumo_bottom[i]) or np.isnan(trend_1d_up_aligned[i]) or 
            np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Tenkan/Kijun cross
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Enter long: TK cross up + price above Kumo + 1d uptrend
            if tk_cross_up and close[i] > kumo_top[i] and trend_1d_up_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross down + price below Kumo + 1d downtrend
            elif tk_cross_down and close[i] < kumo_bottom[i] and trend_1d_down_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when TK cross down OR price breaks below Kumo (invalidates uptrend)
            if tk_cross_down or close[i] < kumo_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when TK cross up OR price breaks above Kumo (invalidates downtrend)
            if tk_cross_up or close[i] > kumo_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals