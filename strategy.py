#!/usr/bin/env python3
# 6h_ichimoku_1d_trend_follow_v1
# Hypothesis: Uses Ichimoku cloud from daily timeframe for trend direction (price above/below cloud) combined with Tenkan-Kijun cross on 6h for entry timing. Works in both bull and bear markets by only taking trades in the direction of the daily trend, reducing whipsaws. Includes volume confirmation to avoid low-liquidity false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Get daily data for Ichimoku cloud (trend filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Tenkan-sen
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max().values + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min().values) / 2
    # Daily Kijun-sen
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max().values + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min().values) / 2
    # Daily Senkou Span A
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    # Daily Senkou Span B
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2)
    
    # Daily cloud boundaries (shifted for alignment - handled by align_htf_to_ltf)
    # Align daily Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(26, 52, vol_ma_period) + 1  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou A and B)
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: Price below cloud or Tenkan-Kijun cross down
            if close[i] < lower_cloud or tenkan[i] < kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above cloud or Tenkan-Kijun cross up
            if close[i] > upper_cloud or tenkan[i] > kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above cloud, Tenkan crosses above Kijun, volume surge
            if (close[i] > upper_cloud and 
                tenkan[i] > kijun[i] and 
                tenkan[i-1] <= kijun[i-1] and  # Cross just happened
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud, Tenkan crosses below Kijun, volume surge
            elif (close[i] < lower_cloud and 
                  tenkan[i] < kijun[i] and 
                  tenkan[i-1] >= kijun[i-1] and  # Cross just happened
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals