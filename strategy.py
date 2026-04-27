#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dTrend_VolumeConfirm
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with 1d trend filter (price above/below Kumo) and volume confirmation. Works in both bull and bear markets by using the 1d Ichimoku cloud as a dynamic trend filter - long only when price is above cloud (bullish regime), short only when price is below cloud (bearish regime). The TK cross provides timely entries within the trend, while volume confirmation reduces false signals. Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_10 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_10 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_10 + min_low_10) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_52 + min_low_52) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Future cloud is plotted 26 periods ahead, but for trend filtering we use current cloud
    # For trend determination: price above both spans = bullish, price below both = bearish
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # 6h Ichimoku for TK cross
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_9 = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (max_high_9 + min_low_9) / 2
    
    max_high_26 = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (max_high_26 + min_low_26) / 2
    
    # TK cross signals
    tk_cross_above = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d Ichimoku (52) + 6h TK components (26) + volume avg (20)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(tk_cross_above[i]) or np.isnan(tk_cross_below[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        tk_above = tk_cross_above[i]
        tk_below = tk_cross_below[i]
        vol_conf = volume_confirm[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        
        if position == 0:
            # Look for entry: TK cross with 1d trend filter (price relative to Kumo) and volume confirmation
            # Long: TK bullish cross AND price above Kumo (bullish regime) AND volume confirmation
            long_condition = tk_above and (close_val > kumo_top_val) and vol_conf
            # Short: TK bearish cross AND price below Kumo (bearish regime) AND volume confirmation
            short_condition = tk_below and (close_val < kumo_bottom_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit conditions:
            # 1. TK bearish cross (tenkan crosses below kijun)
            # 2. Price falls below Kumo bottom (trend change)
            exit_condition = tk_below or (close_val < kumo_bottom_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. TK bullish cross (tenkan crosses above kijun)
            # 2. Price rises above Kumo top (trend change)
            exit_condition = tk_above or (close_val > kumo_top_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0