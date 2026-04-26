#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm
Hypothesis: Trade Ichimoku cloud twists (Tenkan/Kijun cross inside cloud) on 6h with 1d EMA50 trend filter and volume confirmation (1.8x average). Works in bull/bear: long when price above cloud + bullish twist + uptrend; short when price below cloud + bearish twist + downtrend. Targets 12-30 trades/year via strict cloud/twist/confluence requirements.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B: (52-period high + 52-period low) / 2 plotted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components (no extra delay needed as they're based on completed periods)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # Using 1d index for alignment, values are 6h
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: 1.8x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku (52), 1d EMA (50), volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Price relative to cloud
        price_above_cloud = close_val > upper_cloud
        price_below_cloud = close_val < lower_cloud
        
        # Kumo twist detection: Tenkan/Kijun cross inside cloud
        # Bullish twist: Tenkan crosses above Kijun while both are inside cloud
        # Bearish twist: Tenkan crosses below Kijun while both are inside cloud
        if i > start_idx:
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            
            bullish_twist = (tenkan_prev <= kijun_prev and tenkan_val > kijun_val and 
                           tenkan_val > lower_cloud and tenkan_val < upper_cloud and
                           kijun_val > lower_cloud and kijun_val < upper_cloud)
            bearish_twist = (tenkan_prev >= kijun_prev and tenkan_val < kijun_val and 
                           tenkan_val > lower_cloud and tenkan_val < upper_cloud and
                           kijun_val > lower_cloud and kijun_val < upper_cloud)
        else:
            bullish_twist = False
            bearish_twist = False
        
        # Volume confirmation
        volume_spike = volume_val > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: price above cloud + bullish twist + uptrend (close > EMA50) + volume spike
            long_signal = price_above_cloud and bullish_twist and (close_val > ema_50_1d_val) and volume_spike
            # Short: price below cloud + bearish twist + downtrend (close < EMA50) + volume spike
            short_signal = price_below_cloud and bearish_twist and (close_val < ema_50_1d_val) and volume_spike
            
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
            # Exit: price breaks below cloud OR trend reversal (close < EMA50)
            if (close_val < lower_cloud) or (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above cloud OR trend reversal (close > EMA50)
            if (close_val > upper_cloud) or (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0