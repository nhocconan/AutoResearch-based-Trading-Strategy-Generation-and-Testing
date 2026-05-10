#!/usr/bin/env python3
# 6h_1d_Ichimoku_TenkanKijun_CloudFilter_v1
# Hypothesis: 6h strategy using Ichimoku Tenkan/Kijun cross with 1d cloud filter for trend confirmation.
# In bull markets, price above cloud + bullish TK cross = long.
# In bear markets, price below cloud + bearish TK cross = short.
# Uses volume confirmation to filter false breaks.
# Designed for low trade frequency (<30/year) to avoid fee drag in 6h timeframe.
# Works in both bull/bear via cloud as dynamic support/resistance.

name = "6h_1d_Ichimoku_TenkanKijun_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Ichimoku Cloud: top = max(Senkou A, Senkou B), bottom = min(Senkou A, Senkou B)
    # We only need current cloud for filtering
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # TK Cross signals
    tk_cross_above = tenkan_aligned > kijun_aligned  # Bullish cross
    tk_cross_below = tenkan_aligned < kijun_aligned  # Bearish cross
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top_aligned
    price_below_cloud = close < cloud_bottom_aligned
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top_aligned[i]) or
            np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish TK cross + price above cloud + volume surge
            if tk_cross_above[i] and price_above_cloud[i] and volume_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below cloud + volume surge
            elif tk_cross_below[i] and price_below_cloud[i] and volume_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: bearish TK cross OR price drops below cloud
                if tk_cross_below[i] or close[i] < cloud_bottom_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish TK cross OR price rises above cloud
                if tk_cross_above[i] or close[i] > cloud_top_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals