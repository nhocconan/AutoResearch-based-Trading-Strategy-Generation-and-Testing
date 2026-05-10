#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_1d_Adx_Filter
# Hypothesis: Uses 1d Ichimoku cloud for trend direction and 6h ADX for trend strength.
# Long when price is above 1d cloud and 6h ADX > 25; short when price is below 1d cloud and 6h ADX > 25.
# This avoids whipsaws in weak trends and works in both bull and bear markets by following the dominant trend.
# Target: 50-150 total trades over 4 years = 12-37/year.

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
    volume = prices['volume'].values
    
    # 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku components (standard periods: 9, 26, 52)
    def calculate_ichimoku(h, l, c):
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = np.maximum.accumulate(h)
        period9_low = np.minimum.accumulate(l)
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = np.maximum.accumulate(h)
        period26_low = np.minimum.accumulate(l)
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high = np.maximum.accumulate(h)
        period52_low = np.minimum.accumulate(l)
        senkou_b = (period52_high + period52_low) / 2
        
        return tenkan, kijun, senkou_a, senkou_b
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Kumo (cloud) top and bottom
    cloud_top = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # 6h ADX for trend strength
    def calculate_adx(h, l, c, period=14):
        # True Range
        tr1 = h - l
        tr2 = np.abs(h - np.concatenate([[c[0]], c[:-1]]))
        tr3 = np.abs(l - np.concatenate([[c[0]], c[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = h - np.concatenate([[h[0]], h[:-1]])
        down_move = np.concatenate([[l[0]], l[:-1]]) - l
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth_rma(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(arr[:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(arr)):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smooth = smooth_rma(tr, period)
        plus_dm_smooth = smooth_rma(plus_dm, period)
        minus_dm_smooth = smooth_rma(minus_dm, period)
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.where(tr_smooth != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smooth_rma(dx, period)
        
        return adx
    
    adx_6h = calculate_adx(high, low, close, 14)
    
    # Align 1d Ichimoku to 6s timeframe (wait for 1d bar to close)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or \
           np.isnan(adx_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when ADX indicates strong trend (ADX > 25)
        if adx_6h[i] > 25:
            if position == 0:
                # Long: price above cloud
                if close[i] > cloud_top_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price below cloud
                elif close[i] < cloud_bottom_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Long exit: price drops below cloud bottom
                if close[i] < cloud_bottom_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price rises above cloud top
                if close[i] > cloud_top_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Weak trend: exit any position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals