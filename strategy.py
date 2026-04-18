#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v2
Hypothesis: Use 1d Ichimoku cloud to determine trend and 6h Tenkan-Kijun cross for entry timing.
In uptrend (price > 1d cloud), enter long on TK cross up; in downtrend (price < 1d cloud), enter short on TK cross down.
Exit on opposite TK cross or when price crosses the 1d Kijun line. This combines trend filtering with momentum
entries, reducing false signals in chop. Designed for low frequency (10-30 trades/year) to avoid fee drag.
"""

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
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Determine cloud boundaries and trend
    # Cloud top is max(senkou_a, senkou_b), bottom is min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Trend: price above cloud = uptrend, below cloud = downtrend
    # For cloud color: senkou_a > senkou_b = bullish cloud, else bearish
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60  # Need enough data for Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        
        # Determine trend based on price vs cloud
        price_above_cloud = price > cloud_top_val
        price_below_cloud = price < cloud_bottom_val
        
        # TK cross signals
        tk_cross_up = tenkan_val > kijun_val and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_val < kijun_val and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 0:
            # Long: price above cloud (uptrend) and TK cross up
            if price_above_cloud and tk_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud (downtrend) and TK cross down
            elif price_below_cloud and tk_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TK cross down OR price crosses below cloud bottom (trend change)
            if tk_cross_down or price < cloud_bottom_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TK cross up OR price crosses above cloud top (trend change)
            if tk_cross_up or price > cloud_top_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v2"
timeframe = "6h"
leverage = 1.0