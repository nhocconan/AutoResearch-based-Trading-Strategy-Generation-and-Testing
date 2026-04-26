#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_HTFVolSpike_v1
Hypothesis: On 6h timeframe, use Ichimoku TK cross (Tenkan/Kijun) with 1d cloud filter (price above/below cloud) and 1d volume spike confirmation. The cloud acts as dynamic support/resistance, TK cross provides momentum signal, and volume spike confirms institutional interest. Works in bull/bear markets because cloud filter ensures we only trade with the higher timeframe trend (price above cloud = bullish bias, below = bearish bias). Targets 12-35 trades/year via strict entry conditions (TK cross + cloud + volume spike) to minimize fee drag on 6h chart.
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
    volume = prices['volume'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    
    # The cloud is between senkou_a and senkou_b
    # For cloud twist detection, we need previous values
    senkou_a_prev = np.roll(senkou_a, 1)
    senkou_b_prev = np.roll(senkou_b, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    # Get 1d HTF data for multi-timeframe filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku cloud for trend filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # 1d Tenkan-sen (9-period)
    d1_period9_high = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    d1_period9_low = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    d1_tenkan = (d1_period9_high + d1_period9_low) / 2
    
    # 1d Kijun-sen (26-period)
    d1_period26_high = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    d1_period26_low = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    d1_kijun = (d1_period26_high + d1_period26_low) / 2
    
    # 1d Senkou Span A and B
    d1_senkou_a = (d1_tenkan + d1_kijun) / 2
    d1_period52_high = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    d1_period52_low = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    d1_senkou_b = (d1_period52_high + d1_period52_low) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    d1_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, d1_senkou_a)
    d1_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, d1_senkou_b)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    d1_vol_avg = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    d1_volume_spike = df_1d_volume > (2.0 * d1_vol_avg)
    d1_volume_spike_aligned = align_htf_to_ltf(prices, df_1d, d1_volume_spike)
    
    # TK cross signals
    # Bullish TK cross: Tenkan crosses above Kijun
    tk_bullish = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    # Bearish TK cross: Tenkan crosses below Kijun
    tk_bearish = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Cloud filter: price above cloud (bullish) or below cloud (bearish)
    # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # 1d cloud filter (HTF trend)
    d1_cloud_top = np.maximum(d1_senkou_a_aligned, d1_senkou_b_aligned)
    d1_cloud_bottom = np.minimum(d1_senkou_a_aligned, d1_senkou_b_aligned)
    price_above_d1_cloud = close > d1_cloud_top
    price_below_d1_cloud = close < d1_cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Ichimoku calculations (52 periods)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(d1_senkou_a_aligned[i]) or np.isnan(d1_senkou_b_aligned[i]) or
            np.isnan(d1_volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Flat - look for TK cross with cloud and volume confirmation
            # Long: bullish TK cross + price above 1d cloud + 1d volume spike
            long_entry = tk_bullish[i] and price_above_d1_cloud[i] and d1_volume_spike_aligned[i]
            # Short: bearish TK cross + price below 1d cloud + 1d volume spike
            short_entry = tk_bearish[i] and price_below_d1_cloud[i] and d1_volume_spike_aligned[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on bearish TK cross or price breaks below 1d cloud
            exit_condition = tk_bearish[i] or (close[i] < d1_cloud_bottom[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on bullish TK cross or price breaks above 1d cloud
            exit_condition = tk_bullish[i] or (close[i] > d1_cloud_top[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_HTFVolSpike_v1"
timeframe = "6h"
leverage = 1.0