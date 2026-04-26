#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_v2
Hypothesis: 6h Ichimoku Kumo twist (Tenkan/Kijun cross inside cloud) with 1d EMA50 trend filter.
Works in bull/bear: EMA50 defines trend direction, Kumo twist catches reversals within that trend.
Target: 12-30 trades/year (50-120 over 4 years) via strict Kumo twist + trend alignment.
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
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
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
    
    # Kumo twist: Tenkan crosses Kijun while both are inside the cloud
    # Cloud top/bottom (aligned to current time, no look-ahead)
    # Senkou spans are plotted 26 periods ahead, so to get current cloud we look back 26
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Tenkan/Kijun cross signals
    tenkan_prev = np.roll(tenkan, 1)
    kijun_prev = np.roll(kijun, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    tk_cross_up = (tenkan > kijun) & (tenkan_prev <= kijun_prev)
    tk_cross_down = (tenkan < kijun) & (tenkan_prev >= kijun_prev)
    
    # Price inside cloud (both Tenkan and Kijun between cloud boundaries)
    price_in_cloud = (tenkan > cloud_bottom) & (tenkan < cloud_top) & \
                     (kijun > cloud_bottom) & (kijun < cloud_top)
    
    # Kumo twist signals: cross occurs while inside cloud
    kumotwist_long = tk_cross_up & price_in_cloud
    kumotwist_short = tk_cross_down & price_in_cloud
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52) + 1d EMA (50)
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(kumotwist_long[i]) or
            np.isnan(kumotwist_short[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Kumo twist bullish + price above 1d EMA50 (uptrend)
            long_signal = kumotwist_long[i] and (close_val > ema_50_1d_val)
            # Short: Kumo twist bearish + price below 1d EMA50 (downtrend)
            short_signal = kumotwist_short[i] and (close_val < ema_50_1d_val)
            
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
            # Exit: Kumo twist bearish OR price crosses below 1d EMA50
            if kumotwist_short[i] or (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Kumo twist bullish OR price crosses above 1d EMA50
            if kumotwist_long[i] or (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_v2"
timeframe = "6h"
leverage = 1.0