#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Price Action with 1d Trend Filter
Hypothesis: Ichimoku provides robust trend signals; using 1d cloud and TK cross on 6h filters false signals. Works in bull/bear by only trading when price is above/below 1d cloud, reducing whipsaw. Targets 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1dcloud_tk_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkou_a, senkou_b, chikou"""
    n = len(high)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    chikou = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    for i in range(n):
        if i >= 8:
            hh9 = np.max(high[i-8:i+1])
            ll9 = np.min(low[i-8:i+1])
            tenkan[i] = (hh9 + ll9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    for i in range(n):
        if i >= 25:
            hh26 = np.max(high[i-25:i+1])
            ll26 = np.min(low[i-25:i+1])
            kijun[i] = (hh26 + ll26) / 2
    
    # Senkou Span A (Leading Span A): (tenkan + kijun)/2 shifted 26 periods ahead
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + 26
            if idx < n:
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    for i in range(n):
        if i >= 51:
            hh52 = np.max(high[i-51:i+1])
            ll52 = np.min(low[i-51:i+1])
            senkou_b[i+26] = (hh52 + ll52) / 2 if i + 26 < n else np.nan
    
    # Chikou Span (Lagging Span): close shifted 26 periods back
    for i in range(n):
        if i >= 26:
            chikou[i-26] = close[i]
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for Ichimoku
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku for cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, _ = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # 1d Cloud top and bottom
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # ADX for trend strength filter (optional, using 14-period)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        
        atr = np.full(n, np.nan)
        dm_plus_smooth = np.full(n, np.nan)
        dm_minus_smooth = np.full(n, np.nan)
        
        if n >= period:
            atr[period-1] = np.mean(tr[:period])
            dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
            dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
            
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        dx = np.full(n, np.nan)
        for i in range(period, n):
            if atr[i] > 0:
                dx[i] = 100 * np.abs(dm_plus_smooth[i] - dm_minus_smooth[i]) / (dm_plus_smooth[i] + dm_minus_smooth[i])
        
        adx = np.full(n, np.nan)
        if n >= period * 2:
            adx[period*2-1] = np.mean(dx[period:period*2])
            for i in range(period*2, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = 60  # Warmup for Ichimoku
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below cloud OR TK cross down
            if (close[i] < cloud_bottom[i] or 
                (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above cloud OR TK cross up
            if (close[i] > cloud_top[i] or 
                (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price vs cloud + ADX filter
            tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            
            price_above_cloud = close[i] > cloud_top[i]
            price_below_cloud = close[i] < cloud_bottom[i]
            
            # Strong trend filter: ADX > 25
            strong_trend = adx[i] > 25
            
            if tk_cross_up and price_above_cloud and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif tk_cross_down and price_below_cloud and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals