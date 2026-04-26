#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeSpike
Hypothesis: Ichimoku Kumo twist (Senkou Span A/B cross) on 6h with 1d trend filter (price >/<? EMA50) and volume spike >1.8x median. Kumo twist signals trend acceleration, filtered by 1d EMA to avoid counter-trend trades. Volume spike confirms participation. Designed for 6h timeframe: low frequency (~20-40 trades/year), works in bull/bear via trend filter. Uses discrete 0.25 position size.
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
    
    # Load 6h data for Ichimoku calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # Load 1d data for trend filter and volume reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku components on 6h
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Kumo twist signal: Senkou A crosses Senkou B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_prev = np.roll(senkou_a, 1)
    senkou_b_prev = np.roll(senkou_b, 1)
    senkou_a_prev[0] = senkou_a[0]
    senkou_b_prev[0] = senkou_b[0]
    
    bullish_twist = (senkou_a > senkou_b) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a < senkou_b) & (senkou_a_prev >= senkou_b_prev)
    
    # Align Ichimoku signals to 6h (no extra delay as based on completed 6h candles)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_6h, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_6h, bearish_twist.astype(float))
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: volume > 1.8x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.8 * vol_median_20)
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 52 for Senkou B, 50 for 1d EMA, 20 for volume median
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bullish_twist_aligned[i]) or
            np.isnan(bearish_twist_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: bullish Kumo twist + price above 1d EMA50 + volume spike
            long_entry = bullish_twist_aligned[i] and (close_val > ema_50_val) and vol_spike
            # Short: bearish Kumo twist + price below 1d EMA50 + volume spike
            short_entry = bearish_twist_aligned[i] and (close_val < ema_50_val) and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on bearish Kumo twist or price below 1d EMA50
            if bearish_twist_aligned[i] or (close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on bullish Kumo twist or price above 1d EMA50
            if bullish_twist_aligned[i] or (close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0