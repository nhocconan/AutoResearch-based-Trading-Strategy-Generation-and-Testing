#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with Daily Trend Filter
- Uses 6-hour Ichimoku system: TK cross + price above/below cloud
- Confirms trend with 1-day price above/below 50 EMA
- Volume confirmation: 6h volume > 1.5x 20-period average
- Designed for 15-30 trades/year (60-120 total) to minimize fee drift
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkou A/B, chikou."""
    n = len(high)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    for i in range(8, n):
        tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    for i in range(25, n):
        kijun[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + 26
            if idx < n:
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    for i in range(51, n):
        senkou_b[i+26] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average."""
    ema = np.full(len(arr), np.nan)
    if len(arr) < period:
        return ema
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        ema[i] = (arr[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_sma(arr, period):
    """Calculate Simple Moving Average."""
    sma = np.full(len(arr), np.nan)
    if len(arr) < period:
        return sma
    for i in range(period-1, len(arr)):
        sma[i] = np.mean(arr[i-period+1:i+1])
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h Ichimoku data
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Ichimoku components on 6h
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high_6h, low_6h, close_6h)
    
    # Calculate 6h volume moving average (20-period)
    vol_ma_6h = calculate_sma(volume_6h, 20)
    
    # Get 1d data for trend filter (50 EMA)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = calculate_ema(close_1d, 50)
    
    # Align 6h indicators to 6h timeframe
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need sufficient data for Ichimoku (52+26) and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or 
            np.isnan(senkou_a_6h_aligned[i]) or np.isnan(senkou_b_6h_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i]) or np.isnan(ema_50_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 6h volume for current 6h bar
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        vol_confirm = vol_6h_aligned[i] > 1.5 * vol_ma_6h_aligned[i]
        
        # Determine cloud boundaries (senkou A/B)
        cloud_top = np.maximum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        cloud_bottom = np.minimum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        
        # Ichimoku signals
        tk_cross_bull = tenkan_6h_aligned[i] > kijun_6h_aligned[i]
        tk_cross_bear = tenkan_6h_aligned[i] < kijun_6h_aligned[i]
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_6h[i]
        downtrend = close[i] < ema_50_1d_6h[i]
        
        if position == 0:
            # Long: bullish TK cross + price above cloud + uptrend + volume
            if tk_cross_bull and price_above_cloud and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below cloud + downtrend + volume
            elif tk_cross_bear and price_below_cloud and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below cloud OR bearish TK cross
            if price_below_cloud or tk_cross_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above cloud OR bullish TK cross
            if price_above_cloud or tk_cross_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0