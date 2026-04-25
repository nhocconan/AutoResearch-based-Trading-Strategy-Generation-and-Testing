#!/usr/bin/env python3
"""
6h Ichimoku Cloud TK Cross with 1d Cloud Filter and Volume Spike
Hypothesis: Ichimoku Tenkan/Kijun cross (TK) on 6h acts as momentum signal,
while 1d Ichimoku cloud acts as trend filter (price above cloud = bull bias,
below cloud = bear bias). Volume spike confirms institutional participation.
Works in bull markets via long TK crosses above cloud and in bear markets via
short TK crosses below cloud. Uses discrete position sizing (0.25) to limit
drawdown during 2022 crash. Target: 50-150 total trades over 4 years (12-37/year)
on 6h timeframe.
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
    
    # Get 1d data for Ichimoku cloud (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # Calculate Ichimoku components on 6h for TK cross
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_6h_calc = (high_9_6h + low_9_6h) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_6h_calc = (high_26_6h + low_26_6h) / 2
    
    # Calculate 20-period volume MA for volume confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations and volume MA
    start_idx = max(52, 26, 9, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(tenkan_6h_calc[i]) or np.isnan(kijun_6h_calc[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan_1d = tenkan_6h[i]
        kijun_1d = kijun_6h[i]
        senkou_a_1d = senkou_a_6h[i]
        senkou_b_1d = senkou_b_6h[i]
        tenkan_6h_val = tenkan_6h_calc[i]
        kijun_6h_val = kijun_6h_calc[i]
        vol_ma = vol_ma_20[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_1d, senkou_b_1d)
        cloud_bottom = min(senkou_a_1d, senkou_b_1d)
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # TK cross signals
        tk_cross_up = tenkan_6h_val > kijun_6h_val
        tk_cross_down = tenkan_6h_val < kijun_6h_val
        
        if position == 0:
            # Long signal: TK cross up, price above cloud, volume confirmation
            long_signal = tk_cross_up and (curr_close > cloud_top) and volume_confirm
            # Short signal: TK cross down, price below cloud, volume confirmation
            short_signal = tk_cross_down and (curr_close < cloud_bottom) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross down or price closes below cloud
            if tk_cross_down or (curr_close < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross up or price closes above cloud
            if tk_cross_up or (curr_close > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0