#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_V1
Hypothesis: 6h Ichimoku cloud breakout with 1d EMA50 trend filter. Long when price breaks above cloud and 1d EMA50 rising; short when price breaks below cloud and 1d EMA50 falling. Uses discrete position sizing (0.25) to limit drawdown. Targets 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Ichimoku components (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_tenkan + min_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_kijun + min_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    max_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_senkou_b + min_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe (no look-ahead)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # The cloud is between Senkou Span A and Senkou Span B
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(upper_cloud[i]) or 
            np.isnan(lower_cloud[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Long: price breaks above cloud AND 1d EMA50 is rising (bullish trend)
            if price > upper_cloud[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud AND 1d EMA50 is falling (bearish trend)
            elif price < lower_cloud[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below cloud OR 1d EMA50 turns down
            if price < upper_cloud[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud OR 1d EMA50 turns up
            if price > lower_cloud[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_V1"
timeframe = "6h"
leverage = 1.0