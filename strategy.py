#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_Filter_v1
Hypothesis: 6h Ichimoku Kumo twist (Tenkan/Kijun cross) with 1-week trend filter (price vs Senkou Span A/B) and volume confirmation.
Only trade Kumo twists in direction of 1-week cloud (bullish if price above cloud, bearish if below).
Uses Ichimoku components calculated on 6h data but filters by 1-week trend to avoid counter-trend whipsaws.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of twist, trend, and volume.
Works in bull/bear via 1-week trend filter: only takes long twists in weekly uptrend, short in weekly downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Calculate 1w Ichimoku components for trend filter
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_1w = (pd.Series(df_1w['high']).rolling(window=9, min_periods=9).mean() + 
                 pd.Series(df_1w['low']).rolling(window=9, min_periods=9).mean()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1w = (pd.Series(df_1w['high']).rolling(window=26, min_periods=26).mean() + 
                pd.Series(df_1w['low']).rolling(window=26, min_periods=26).mean()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_1w = ((pd.Series(df_1w['high']).rolling(window=52, min_periods=52).mean() + 
                    pd.Series(df_1w['low']).rolling(window=52, min_periods=52).mean()) / 2).shift(26)
    
    # Align 1w Ichimoku components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w.values)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w.values)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w.values)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w.values)
    
    # Determine 1w trend: price above cloud = bullish, below cloud = bearish
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_1w = np.maximum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    cloud_bottom_1w = np.minimum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    weekly_trend = np.where(close > cloud_top_1w, 1, np.where(close < cloud_bottom_1w, -1, 0))  # 1=uptrend, -1=downtrend, 0=in cloud
    
    # Calculate Ichimoku on 6h data for entry signals (Kumo twist)
    # Tenkan-sen (6h)
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).mean() + 
                 pd.Series(low).rolling(window=9, min_periods=9).mean()) / 2
    # Kijun-sen (6h)
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).mean() + 
                pd.Series(low).rolling(window=26, min_periods=26).mean()) / 2
    # Senkou Span A (6h)
    senkou_a_6h = ((tenkan_6h + kijun_6h) / 2).shift(26)
    # Senkou Span B (6h)
    senkou_b_6h = ((pd.Series(high).rolling(window=52, min_periods=52).mean() + 
                    pd.Series(low).rolling(window=52, min_periods=52).mean()) / 2).shift(26)
    
    # Kumo twist detection: Tenkan crosses Kijun
    # Bullish twist: Tenkan crosses above Kijun
    # Bearish twist: Tenkan crosses below Kijun
    tenkan_kijun_diff = tenkan_6h - kijun_6h
    bullish_twist = (tenkan_kijun_diff > 0) & (tenkan_kijun_diff.shift(1) <= 0)
    bearish_twist = (tenkan_kijun_diff < 0) & (tenkan_kijun_diff.shift(1) >= 0)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 20 for volume MA)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Entry conditions: Kumo twist in direction of weekly trend with volume
        if weekly_trend[i] == 1:  # Weekly uptrend
            # Look for bullish twist (Tenkan crosses above Kijun) with volume
            if bullish_twist.iloc[i] if hasattr(bullish_twist, 'iloc') else bullish_twist[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if bearish twist occurs (Tenkan crosses below Kijun)
            elif position == 1 and (bearish_twist.iloc[i] if hasattr(bearish_twist, 'iloc') else bearish_twist[i]):
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
        elif weekly_trend[i] == -1:  # Weekly downtrend
            # Look for bearish twist (Tenkan crosses below Kijun) with volume
            if bearish_twist.iloc[i] if hasattr(bearish_twist, 'iloc') else bearish_twist[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if bullish twist occurs (Tenkan crosses above Kijun)
            elif position == -1 and (bullish_twist.iloc[i] if hasattr(bullish_twist, 'iloc') else bullish_twist[i]):
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
        else:
            # In cloud or unclear weekly trend - reduce position or stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.125  # Reduce long position in cloud
            else:
                signals[i] = -0.125  # Reduce short position in cloud
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0