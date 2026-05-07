#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend_Volume
Hypothesis: Ichimoku Tenkan/Kijun cross with cloud filter from 1d, combined with 1d trend and volume confirmation.
Works in both bull and bear markets by using cloud color (bullish/bearish) as regime filter.
Entry: TK cross in direction of 1d trend, price above/below cloud accordingly, volume > 1.5x average.
Exit: TK cross in opposite direction or price crosses Kijun line.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2  # Tenkan-sen (Conversion Line)
    
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2  # Kijun-sen (Base Line)
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B)
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span) - not used for signals but for cloud
    # Cloud is between Senkou A and Senkou B
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need sufficient warmup for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        lower_cloud = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Cloud color: bullish when Senkou A > Senkou B
        cloud_bullish = senkou_a_6h[i] > senkou_b_6h[i]
        cloud_bearish = senkou_a_6h[i] < senkou_b_6h[i]
        
        # Determine market regime from 1-day EMA50
        uptrend_regime = close[i] > ema_50_1d_aligned[i]
        downtrend_regime = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        # TK Cross signals
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 0:
            # Long: TK cross up in uptrend regime, price above cloud, bullish cloud, volume confirmation
            long_entry = tk_cross_up and uptrend_regime and (close[i] > upper_cloud) and cloud_bullish and volume_confirm
            # Short: TK cross down in downtrend regime, price below cloud, bearish cloud, volume confirmation
            short_entry = tk_cross_down and downtrend_regime and (close[i] < lower_cloud) and cloud_bearish and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross down or price crosses below Kijun or cloud turns bearish
            if tk_cross_down or (close[i] < kijun_6h[i]) or (not cloud_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross up or price crosses above Kijun or cloud turns bullish
            if tk_cross_up or (close[i] > kijun_6h[i]) or (not cloud_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals