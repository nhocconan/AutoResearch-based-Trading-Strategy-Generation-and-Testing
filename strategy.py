#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeSpike_v1
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun cross with 1d cloud filter (price above/below cloud) and volume confirmation provides robust trend signals. Works in bull markets (long when price above cloud + TK cross up) and bear markets (short when price below cloud + TK cross down). Uses discrete sizing (0.0, ±0.25) and volume spike (>2x avg) to reduce false signals. Targets 50-150 trades over 4 years (12-37/year) for optimal 6h frequency.
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
    
    # Get 1d data for HTF cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need enough for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Get 1d Ichimoku cloud for HTF filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    df_1d_period9_high = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    df_1d_period9_low = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    df_1d_tenkan = (df_1d_period9_high + df_1d_period9_low) / 2
    
    # 1d Kijun-sen (26-period)
    df_1d_period26_high = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    df_1d_period26_low = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    df_1d_kijun = (df_1d_period26_high + df_1d_period26_low) / 2
    
    # 1d Senkou Span A
    df_1d_senkou_a = (df_1d_tenkan + df_1d_kijun) / 2
    
    # 1d Senkou Span B (52-period)
    df_1d_period52_high = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    df_1d_period52_low = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    df_1d_senkou_b = (df_1d_period52_high + df_1d_period52_low) / 2
    
    # Align 1d cloud to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_senkou_a)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d_senkou_b)
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods (52) and volume MA (20)
    start_idx = max(52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        # Determine 1d cloud boundaries
        upper_cloud_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price vs cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # 1d cloud filter: price should be above/below 1d cloud for alignment
        price_above_1d_cloud = close[i] > upper_cloud_1d
        price_below_1d_cloud = close[i] < lower_cloud_1d
        
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        
        if position == 0:
            # Long: TK cross up + price above 6h cloud + price above 1d cloud + volume
            long_signal = tk_cross_up and price_above_cloud and price_above_1d_cloud and vol_confirmed
            
            # Short: TK cross down + price below 6h cloud + price below 1d cloud + volume
            short_signal = tk_cross_down and price_below_cloud and price_below_1d_cloud and vol_confirmed
            
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
            # Exit: TK cross down OR price closes below 6h cloud
            if tk_cross_down or close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price closes above 6h cloud
            if tk_cross_up or close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0