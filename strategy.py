#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_1d_Adx_Filter
# Hypothesis: Uses Ichimoku cloud on 1d timeframe for trend direction and ADX(14) on 6h for trend strength.
# Long when price above Kumo cloud, Tenkan > Kijun, and ADX > 25.
# Short when price below Kumo cloud, Tenkan < Kijun, and ADX > 25.
# Ichimoku provides dynamic support/resistance and trend direction; ADX filters for strong trends only.
# Designed for 6h to achieve 12-37 trades/year, works in both bull and bear markets by following strong trends.

name = "6h_Ichimoku_Cloud_Trend_1d_Adx_Filter"
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
    
    # 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    def calculate_ichimoku(h, l, c):
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = np.maximum.accumulate(h)
        period9_low = np.minimum.accumulate(l)
        tenkan = (np.concatenate([np.full(8, np.nan), (period9_high[8:] + period9_low[8:]) / 2]))
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = np.maximum.accumulate(h)
        period26_low = np.minimum.accumulate(l)
        kijun = (np.concatenate([np.full(25, np.nan), (period26_high[25:] + period26_low[25:]) / 2]))
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
        period52_high = np.maximum.accumulate(h)
        period52_low = np.minimum.accumulate(l)
        senkou_b = (np.concatenate([np.full(51, np.nan), (period52_high[51:] + period52_low[51:]) / 2]))
        
        # Chikou Span (Lagging Span): close shifted 26 periods behind
        chikou = np.concatenate([np.full(26, np.nan), c[:-26]])
        
        return tenkan, kijun, senkou_a, senkou_b, chikou
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Kumo cloud boundaries (Senkou Span A and B)
    # Kumo top is the higher of Senkou A and Senkou B
    # Kumo bottom is the lower of Senkou A and Senkou B
    kumo_top = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # ADX(14) on 6h for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.concatenate([[high[0]], high[:-1]])) > 
                          (np.concatenate([[low[0]], low[:-1]]) - low), 
                          np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0), 0)
        dm_minus = np.where((np.concatenate([[low[0]], low[:-1]]) - low) > 
                           (high - np.concatenate([[high[0]], high[:-1]])), 
                           np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0), 0)
        
        # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            alpha = 1.0 / period
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.mean(data[:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilders_smoothing(tr, period)
        dm_plus_smooth = wilders_smoothing(dm_plus, period)
        dm_minus_smooth = wilders_smoothing(dm_minus, period)
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / atr
        minus_di = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_6h = calculate_adx(high, low, close, 14)
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for Ichimoku (52 periods) and ADX
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or \
           np.isnan(adx_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX threshold for trend strength
        if adx_6h[i] < 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above Kumo cloud, Tenkan > Kijun
            if close[i] > kumo_top_aligned[i] and tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below Kumo cloud, Tenkan < Kijun
            elif close[i] < kumo_bottom_aligned[i] and tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Kumo cloud or Tenkan < Kijun
            if close[i] < kumo_bottom_aligned[i] or tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Kumo cloud or Tenkan > Kijun
            if close[i] > kumo_top_aligned[i] or tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals