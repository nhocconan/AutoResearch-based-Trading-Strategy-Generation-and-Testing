#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter
Hypothesis: Ichimoku cloud twist (TK cross) on 6h with 1d trend filter (price > 1d EMA50) and volume confirmation.
Works in bull: TK cross up + price above cloud + 1d uptrend + volume spike.
Works in bear: TK cross down + price below cloud + 1d downtrend + volume spike.
Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross).
Volume filter ensures institutional participation. 1d EMA50 avoids counter-trend trades.
Target: 20-40 trades/year on BTC/ETH. Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).mean().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).mean().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).mean().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).mean().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).mean().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).mean().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align Ichimoku components (no extra delay needed as they are concurrent)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # self-align for same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for Ichimoku (52 periods) and 1d EMA50
    start_idx = max(100, 52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_trend = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        # Kumo (cloud) boundaries: Senkou Span A and B
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Flat - look for TK cross in direction of 1d trend with volume confirmation
            # TK Cross: Tenkan crosses Kijun
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan_val > kijun_val)
            tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan_val < kijun_val)
            
            # Long: TK cross up + price above cloud + 1d uptrend + volume spike
            long_entry = tk_cross_up and (close_val > upper_cloud) and (close_val > ema_trend) and vol_spike
            # Short: TK cross down + price below cloud + 1d downtrend + volume spike
            short_entry = tk_cross_down and (close_val < lower_cloud) and (close_val < ema_trend) and vol_spike
            
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
            # Long - exit on TK cross down or price breaks below cloud
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan_val < kijun_val)
            exit_condition = tk_cross_down or (close_val < lower_cloud)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on TK cross up or price breaks above cloud
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan_val > kijun_val)
            exit_condition = tk_cross_up or (close_val > upper_cloud)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0