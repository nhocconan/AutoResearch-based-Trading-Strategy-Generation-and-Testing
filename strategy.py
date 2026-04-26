#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_12hTrend_v1
Hypothesis: On 6h timeframe, Ichimoku TK cross (Tenkan/Kijun) with price above/below cloud from 12h timeframe as trend filter. 
Only take long when TK cross bullish and price > 12h cloud; short when TK cross bearish and price < 12h cloud. 
Add volume confirmation (1.5x median) to avoid whipsaws. Designed for low trade frequency (~20-40/year) to minimize fee drag 
while capturing medium-term trends in both bull and bear markets via multi-timeframe alignment.
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
    
    # Get 12h data for HTF Ichimoku cloud
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # need 26*2 for Senkou B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h (Tenkan, Kijun, Chikou)
    # Tenkan-sen: (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Calculate Ichimoku cloud on 12h (Senkou Span A/B)
    # Senkou Span A: (Tenkan + Kijun)/2 shifted 26 periods ahead
    # Senkou Span B: (52-period high + 52-period low)/2 shifted 26 periods ahead
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    # Tenkan 12h: 9-period
    tenkan_12h = (pd.Series(df_12h_high).rolling(window=9, min_periods=9).max() + 
                  pd.Series(df_12h_low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun 12h: 26-period
    kijun_12h = (pd.Series(df_12h_high).rolling(window=26, min_periods=26).max() + 
                 pd.Series(df_12h_low).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A
    senkou_a = ((tenkan_12h + kijun_12h) / 2)
    # Senkou Span B: 52-period
    senkou_b = (pd.Series(df_12h_high).rolling(window=52, min_periods=52).max() + 
                pd.Series(df_12h_low).rolling(window=52, min_periods=52).min()) / 2
    
    # Align all indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a.values, additional_delay_bars=26)  # SSA is already shifted 26, need extra delay for alignment
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b.values, additional_delay_bars=26)  # SSB same
    
    # Volume confirmation: 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods (26, 52) and volume median (20)
    start_idx = max(26, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_median[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        # Determine cloud boundaries (top = max(SSA, SSB), bottom = min(SSA, SSB))
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross signals
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        # Price relative to cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        if position == 0:
            # Long: bullish TK cross AND price above 12h cloud AND volume spike
            long_signal = tk_bullish and price_above_cloud and (volume_val > 1.5 * vol_median_val)
            # Short: bearish TK cross AND price below 12h cloud AND volume spike
            short_signal = tk_bearish and price_below_cloud and (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long until TK cross turns bearish OR price breaks below cloud
            signals[i] = 0.25
            if tk_bearish or close_val < cloud_bottom:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short until TK cross turns bullish OR price breaks above cloud
            signals[i] = -0.25
            if tk_bullish or close_val > cloud_top:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_12hTrend_v1"
timeframe = "6h"
leverage = 1.0