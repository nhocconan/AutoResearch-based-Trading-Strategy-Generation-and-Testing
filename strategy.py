#!/usr/bin/env python3
# 6h_1d1w_ichimoku_cloud_v1
# Hypothesis: Ichimoku Cloud on daily chart provides strong support/resistance zones.
# Price above/below cloud indicates trend direction from higher timeframe.
# Tenkan-sen/Kijun-sun cross on 6h chart provides entry signals in direction of daily cloud.
# Weekly trend filter ensures alignment with major trend.
# Works in both bull/bear markets by following higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d1w_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    n = len(high)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.zeros(n)
    period9_low = np.zeros(n)
    for i in range(n):
        if i >= 8:
            period9_high[i] = np.max(high[i-8:i+1])
            period9_low[i] = np.min(low[i-8:i+1])
        else:
            period9_high[i] = np.nan
            period9_low[i] = np.nan
    
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.zeros(n)
    period26_low = np.zeros(n)
    for i in range(n):
        if i >= 25:
            period26_high[i] = np.max(high[i-25:i+1])
            period26_low[i] = np.min(low[i-25:i+1])
        else:
            period26_high[i] = np.nan
            period26_low[i] = np.nan
    
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.zeros(n)
    period52_low = np.zeros(n)
    for i in range(n):
        if i >= 51:
            period52_high[i] = np.max(high[i-51:i+1])
            period52_low[i] = np.min(low[i-51:i+1])
        else:
            period52_high[i] = np.nan
            period52_low[i] = np.nan
    
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Determine cloud (Senkou Span A and B)
    # In Ichimoku, the cloud is the area between Senkou Span A and B
    # Senkou Span A is plotted 26 periods ahead, Senkou Span B is plotted 26 periods ahead
    # For simplicity, we use current values and consider price above/both above cloud when price > max(A,B)
    # and below cloud when price < min(A,B)
    cloud_top = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w, dtype=float)
    ema_1w[0] = close_1w[0]
    alpha = 2.0 / (20 + 1)
    for i in range(1, len(close_1w)):
        ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align weekly EMA to 6h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Tenkan/Kijun cross on 6h chart
    period9_high = np.zeros(n)
    period9_low = np.zeros(n)
    period26_high = np.zeros(n)
    period26_low = np.zeros(n)
    
    for i in range(n):
        if i >= 8:
            period9_high[i] = np.max(high[i-8:i+1])
            period9_low[i] = np.min(low[i-8:i+1])
        else:
            period9_high[i] = np.nan
            period9_low[i] = np.nan
            
        if i >= 25:
            period26_high[i] = np.max(high[i-25:i+1])
            period26_low[i] = np.min(low[i-25:i+1])
        else:
            period26_high[i] = np.nan
            period26_low[i] = np.nan
    
    tenkan_6h = (period9_high + period9_low) / 2
    kijun_6h = (period26_high + period26_low) / 2
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine if price is above or below daily cloud
        price_above_cloud = close[i] > cloud_top_aligned[i]
        price_below_cloud = close[i] < cloud_bottom_aligned[i]
        
        # Volume confirmation: current volume > 1.2x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.2
        
        # Tenkan/Kijun cross on 6h chart
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Tenkan/Kijun cross down OR price goes below cloud
            if tk_cross_down or price_below_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan/Kijun cross up OR price goes above cloud
            if tk_cross_up or price_above_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above daily cloud + TK cross up + weekly uptrend + volume
            if price_above_cloud and tk_cross_up and weekly_uptrend and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price below daily cloud + TK cross down + weekly downtrend + volume
            elif price_below_cloud and tk_cross_down and weekly_downtrend and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals