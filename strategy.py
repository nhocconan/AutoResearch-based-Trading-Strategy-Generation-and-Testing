#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Ichimoku cloud breakout on 6h with 1w trend filter (price above/below weekly cloud) and volume confirmation captures medium-term trends while avoiding whipsaws. The Ichimoku system provides dynamic support/resistance via the cloud, and weekly trend alignment ensures we only trade in the direction of the higher timeframe momentum. Volume confirmation filters false breakouts. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for breakout signals)
    
    # Calculate 1w Ichimoku cloud for trend filter
    if len(df_1w) < period_senkou_b:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Tenkan-sen
    max_high_tenkan_1w = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan_1w = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1w = (max_high_tenkan_1w + min_low_tenkan_1w) / 2
    
    # 1w Kijun-sen
    max_high_kijun_1w = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun_1w = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1w = (max_high_kijun_1w + min_low_kijun_1w) / 2
    
    # 1w Senkou Span A
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # 1w Senkou Span B
    max_high_senkou_b_1w = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b_1w = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1w = ((max_high_senkou_b_1w + min_low_senkou_b_1w) / 2)
    
    # Align 1w Ichimoku components to 6h timeframe (cloud edges)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # The cloud is between Senkou Span A and B
    # Top of cloud = max(Senkou A, Senkou B)
    # Bottom of cloud = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    cloud_bottom = np.minimum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    
    # Volume spike detection (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(period_senkou_b + 26, 50, 20)  # Senkou B needs 52 periods + 26 shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter: price above/below weekly cloud
        price_above_weekly_cloud = close[i] > cloud_top[i]
        price_below_weekly_cloud = close[i] < cloud_bottom[i]
        
        # Ichimoku breakout conditions
        # Bullish breakout: price breaks above cloud + Tenkan > Kijun (bullish momentum)
        bullish_breakout = (close[i] > cloud_top[i] and 
                           tenkan[i] > kijun[i] and 
                           volume_spike[i])
        
        # Bearish breakout: price breaks below cloud + Tenkan < Kijun (bearish momentum)
        bearish_breakout = (close[i] < cloud_bottom[i] and 
                           tenkan[i] < kijun[i] and 
                           volume_spike[i])
        
        # Long logic: bullish breakout + price above weekly cloud (uptrend alignment)
        if bullish_breakout and price_above_weekly_cloud:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: bearish breakout + price below weekly cloud (downtrend alignment)
        elif bearish_breakout and price_below_weekly_cloud:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite side of cloud or Tenkan/Kijun cross reverses
        elif position == 1 and (close[i] < cloud_bottom[i] or tenkan[i] < kijun[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > cloud_top[i] or tenkan[i] > kijun[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0