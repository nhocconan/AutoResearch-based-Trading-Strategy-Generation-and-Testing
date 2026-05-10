#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_1d_Adx_Filter
# Hypothesis: 6-hour entries based on 1-day Ichimoku cloud breakouts with ADX trend strength filter and volume confirmation.
# The Ichimoku cloud provides dynamic support/resistance; price above/below cloud indicates trend direction.
# ADX > 25 ensures we only trade in strong trends, avoiding choppy markets.
# Volume > 1.5x 20-period average confirms breakout strength.
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
    volume = prices['volume'].values
    
    # 1-day data for Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    def calculate_ichimoku(high, low, close):
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = np.maximum.accumulate(high)
        period9_low = np.minimum.accumulate(low)
        # Fix for proper windowed calculation
        tenkan_sen = np.full_like(high, np.nan)
        kijun_sen = np.full_like(high, np.nan)
        senkou_span_a = np.full_like(high, np.nan)
        senkou_span_b = np.full_like(high, np.nan)
        
        for i in range(len(high)):
            if i >= 8:  # 9 periods
                tenkan_sen[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
            if i >= 25:  # 26 periods
                kijun_sen[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
            if i >= 51:  # 52 periods for Senkou B
                senkou_span_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
        
        # Senkou Span A: (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        # But we'll calculate current values and let alignment handle timing
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Chikou Span: Close shifted -22 periods (not needed for our logic)
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
    
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # ADX calculation (14 periods)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.mean(data[:period])
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = WilderSmoothing(tr, period)
        dm_plus_smooth = WilderSmoothing(dm_plus, period)
        dm_minus_smooth = WilderSmoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = WilderSmoothing(dx, period)
        
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align 1-day indicators to 6h timeframe (wait for 1d bar to close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or \
           np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud, strong trend (ADX > 25), strong volume
            if close[i] > cloud_top_aligned[i] and adx_aligned[i] > 25 and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, strong trend (ADX > 25), strong volume
            elif close[i] < cloud_bottom_aligned[i] and adx_aligned[i] > 25 and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below cloud bottom or trend weakens (ADX < 20)
            if close[i] < cloud_bottom_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud top or trend weakens (ADX < 20)
            if close[i] > cloud_top_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals