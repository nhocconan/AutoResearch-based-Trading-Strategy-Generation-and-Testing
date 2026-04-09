#!/usr/bin/env python3
# 6h_ichimoku_cloud_breakout_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d HTF for trend direction and 6h Tenkan/Kijun cross for entry timing.
# Enters long when price is above 1d Ichimoku cloud (bullish trend) AND 6h Tenkan crosses above Kijun.
# Enters short when price is below 1d Ichimoku cloud (bearish trend) AND 6h Tenkan crosses below Kijun.
# Exits when price re-enters the 1d cloud or Tenkan/Kijun cross reverses.
# Uses discrete sizing (0.25) to limit fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
# Ichimoku cloud provides dynamic support/resistance that adapts to volatility, working in both bull and bear markets.
# The 1d cloud filters counter-trend noise on 6h, while TK cross provides timely entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # 6h Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Multi-timeframe: 1d Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    high_1d_tenkan = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_1d_tenkan = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_1d_tenkan + low_1d_tenkan) / 2
    
    # 1d Kijun-sen (26-period)
    high_1d_kijun = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_1d_kijun = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_1d_kijun + low_1d_kijun) / 2
    
    # 1d Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B (Leading Span B): (52-period high + low) / 2 plotted 26 periods ahead
    high_1d_senkou_b = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_1d_senkou_b = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_1d_senkou_b + low_1d_senkou_b) / 2
    
    # Align 1d Ichimoku components to 6h timeframe (wait for 1d close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # The cloud is the area between Senkou A and Senkou B
    # Top of cloud = max(Senkou A, Senkou B)
    # Bottom of cloud = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or i < 1 or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # 6h Tenkan/Kijun cross (using previous bar to avoid look-ahead)
        tenkan_prev = tenkan[i-1]
        kijun_prev = kijun[i-1]
        tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan[i] > kijun[i])
        tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan[i] < kijun[i])
        
        # Price relative to 1d cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 1:  # Long position
            # Exit: price re-enters cloud or TK cross turns bearish
            if not price_above_cloud or tk_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters cloud or TK cross turns bullish
            if not price_below_cloud or tk_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for TK cross with cloud filter
            bullish_setup = tk_cross_up and price_above_cloud
            bearish_setup = tk_cross_down and price_below_cloud
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals