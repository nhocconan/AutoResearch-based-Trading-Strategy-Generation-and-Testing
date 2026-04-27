#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter
Hypothesis: 6h Ichimoku cloud twist (Tenkan/Kijun cross) with 1d trend filter (price > weekly EMA50) and volume confirmation.
Ichimoku provides dynamic support/resistance via cloud and momentum via TK cross. Weekly trend filter ensures we only trade
in the higher timeframe direction, reducing false signals. Volume confirmation adds conviction to breakouts.
Works in both bull and bear markets by aligning with weekly trend and using cloud as dynamic filter.
Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/SOL.
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
    
    # Calculate 1d Ichimoku components for cloud and TK cross
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need enough for Senkou B (52 periods)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate weekly EMA50 for trend filter (more stable than daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.8 * 24-period average (6h * 4 = 1 day)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for Ichimoku calculation (52 periods for Senkou B)
    start_idx = max(100, 52)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        # Determine cloud color and position
        # Cloud top = max(senkou_a, senkou_b), Cloud bottom = min(senkou_a, senkou_b)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Bullish cloud: senkou_a > senkou_b (green cloud)
        # Bearish cloud: senkou_a < senkou_b (red cloud)
        bullish_cloud = senkou_a_val > senkou_b_val
        
        # Price above/below cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # TK cross signals
        # Bullish TK cross: Tenkan crosses above Kijun
        # Bearish TK cross: Tenkan crosses below Kijun
        bullish_tk_cross = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1]) if i > 0 else False
        bearish_tk_cross = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1]) if i > 0 else False
        
        if position == 0:
            # Flat - look for entry
            # Long: bullish TK cross AND price above cloud (or breaking above) AND weekly trend up AND volume spike
            # Short: bearish TK cross AND price below cloud (or breaking below) AND weekly trend down AND volume spike
            long_condition = bullish_tk_cross and price_above_cloud and (close_val > ema_trend) and vol_spike
            short_condition = bearish_tk_cross and price_below_cloud and (close_val < ema_trend) and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below cloud bottom OR bearish TK cross
            if price_below_cloud or bearish_tk_cross:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above cloud top OR bullish TK cross
            if price_above_cloud or bullish_tk_cross:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0