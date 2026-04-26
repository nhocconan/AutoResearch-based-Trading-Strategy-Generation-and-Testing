#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeConfirm_v2
Hypothesis: On 6h timeframe, Ichimoku TK cross (Tenkan/Kijun) with 1d cloud filter (price above cloud for long, below for short) and volume confirmation (>1.5x avg) provides robust trend-following signals. The Ichimoku system identifies trend direction and momentum, while the daily cloud acts as a dynamic support/resistance filter. Volume confirmation ensures institutional participation. Works in bull markets (long when price > 1d cloud + TK cross up) and bear markets (short when price < 1d cloud + TK cross down). Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 trades over 4 years (12-37/year) for optimal 6h frequency.
"""

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
    
    # Get daily data for Ichimoku components (needed for Tenkan/Kijun/Senkou Span)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need enough for Senkou Span B (52 periods)
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume ratio (current / 20-period average) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku warmup + volume MA
    start_idx = max(52, 20)  # Senkou B needs 52 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        vol_confirmed = vol_ratio[i] > 1.5  # volume at least 1.5x average
        
        # TK cross conditions
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        if position == 0:
            # Long: price above cloud + TK cross up + volume
            long_signal = (close[i] > upper_cloud and 
                          tk_cross_up and 
                          vol_confirmed)
            
            # Short: price below cloud + TK cross down + volume
            short_signal = (close[i] < lower_cloud and 
                           tk_cross_down and 
                           vol_confirmed)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below cloud OR TK cross down
            if close[i] < lower_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above cloud OR TK cross up
            if close[i] > upper_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0