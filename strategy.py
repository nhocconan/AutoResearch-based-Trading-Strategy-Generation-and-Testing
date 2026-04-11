#!/usr/bin/env python3
"""
6h_1d_ichimoku_cloud_filter_v1
Strategy: 6h Ichimoku with 1d cloud filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses Ichimoku cloud (Tenkan/Kijun/Senkou) on 6h for entry signals, filtered by 1d Ichimoku cloud color (bullish/bearish). In bullish 1d cloud, only take long signals; in bearish 1d cloud, only take short signals. This avoids counter-trend trades and improves win rate in both bull and bear markets by aligning with higher timeframe trend via cloud color. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = pd.Series(close).shift(-period_kijun)  # Will handle alignment via look-ahead avoidance
    
    # Align 6h Ichimoku components to avoid look-ahead
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b.values)
    # Chikou is not used for entry to avoid look-ahead
    
    # 1d Ichimoku for cloud color (trend filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan-sen
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    # 1d Kijun-sen
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B
    high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b_1d = (high_senkou_b_1d + low_senkou_b_1d) / 2
    
    # Align 1d Ichimoku components
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need 52 periods for Senkou B
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # 6h Ichimoku signals
        # Tenkan/Kijun cross
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # 1d cloud color (trend filter)
        # Bullish cloud: Senkou A > Senkou B
        # Bearish cloud: Senkou A < Senkou B
        bullish_cloud_1d = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]
        bearish_cloud_1d = senkou_a_1d_aligned[i] < senkou_b_1d_aligned[i]
        
        # Long: TK cross up + price above cloud + bullish 1d cloud
        long_signal = tk_cross_up and price_above_cloud and bullish_cloud_1d
        
        # Short: TK cross down + price below cloud + bearish 1d cloud
        short_signal = tk_cross_down and price_below_cloud and bearish_cloud_1d
        
        # Exit when TK cross reverses or price enters cloud
        exit_long = position == 1 and (tk_cross_down or not price_above_cloud)
        exit_short = position == -1 and (tk_cross_up or not price_below_cloud)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals