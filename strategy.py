#!/usr/bin/env python3
# 6H_Ichimoku_Cloud_Trend_Follow
# Hypothesis: Use Ichimoku cloud on 1d for trend direction and 6h for entry timing.
# Long when: price above 1d Kumo cloud + Tenkan > Kijun on 6h + price near 6m Kijun (support) + volume confirmation.
# Short when: price below 1d Kumo cloud + Tenkan < Kijun on 6h + price near 6h Kijun (resistance) + volume confirmation.
# Works in bull/bear by following higher timeframe cloud trend and using lower timeframe for pullback entries.
# Target: 15-25 trades/year per symbol.

name = "6H_Ichimoku_Cloud_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = high_s.rolling(window=9, min_periods=9).max()
    period9_low = low_s.rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = high_s.rolling(window=26, min_periods=26).max()
    period26_low = low_s.rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = high_s.rolling(window=52, min_periods=52).max()
    period52_low = low_s.rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    # Daily trend filter using Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku components
    close_1d_s = pd.Series(close_1d)
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    
    # Tenkan-sen 1d
    tenkan_1d = (high_1d_s.rolling(window=9, min_periods=9).max() + 
                 low_1d_s.rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen 1d
    kijun_1d = (high_1d_s.rolling(window=26, min_periods=26).max() + 
                low_1d_s.rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A 1d
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B 1d
    senkou_b_1d = ((high_1d_s.rolling(window=52, min_periods=52).max() + 
                    low_1d_s.rolling(window=52, min_periods=52).min()) / 2)
    
    # Kumo cloud boundaries (Senkou Span A and B shifted forward 26 periods)
    # For trend determination, we use current cloud (Senkou A/B values)
    # Price above cloud: bullish, below cloud: bearish
    # We'll use the average of Senkou A and B as cloud center for simplicity
    cloud_top_1d = np.maximum(senkou_a_1d.values, senkou_b_1d.values)
    cloud_bottom_1d = np.minimum(senkou_a_1d.values, senkou_b_1d.values)
    cloud_mid_1d = (cloud_top_1d + cloud_bottom_1d) / 2
    
    # Determine if price is above or below 1d cloud
    price_above_cloud = close_1d > cloud_top_1d
    price_below_cloud = close_1d < cloud_bottom_1d
    
    # Align 1d cloud signals to 6h
    price_above_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_above_cloud.astype(float))
    price_below_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_below_cloud.astype(float))
    
    # Volume confirmation (20-period average)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(vol_ma[i]) or
            np.isnan(price_above_cloud_aligned[i]) or np.isnan(price_below_cloud_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price position relative to 6h Kijun (support/resistance)
        price_vs_kijun = close[i] - kijun[i]
        near_kijun = abs(price_vs_kijun) < (kijun[i] * 0.015)  # within 1.5%
        
        # Tenkan/Kijun cross
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.3
        
        # 1d cloud trend
        above_1d_cloud = price_above_cloud_aligned[i] > 0.5
        below_1d_cloud = price_below_cloud_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price above 1d cloud + bullish TK cross + near Kijun support + volume
            if above_1d_cloud and tenkan_above_kijun and near_kijun and volume_confirm and price_vs_kijun >= 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price below 1d cloud + bearish TK cross + near Kijun resistance + volume
            elif below_1d_cloud and tenkan_below_kijun and near_kijun and volume_confirm and price_vs_kijun <= 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish TK cross or price moves above cloud or too far from Kijun
            if tenkan_below_kijun or not above_1d_cloud or price_vs_kijun > (kijun[i] * 0.025):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross or price moves below cloud or too far from Kijun
            if tenkan_above_kijun or not below_1d_cloud or price_vs_kijun < (-kijun[i] * 0.025):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals