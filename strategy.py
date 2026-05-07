#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 12h data for Ichimoku and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    # Ichimoku components (standard parameters: 9, 26, 52)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, bullish trend, volume
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] and
                close[i] > cloud_top and ema_50_6h[i] > ema_50_6h[i-1] and vol_condition):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, bearish trend, volume
            elif (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] and
                  close[i] < cloud_bottom and ema_50_6h[i] < ema_50_6h[i-1] and vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Tenkan crosses below Kijun or price falls below cloud
            if (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]) or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses above Kijun or price rises above cloud
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]) or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku TK cross with cloud filter and 12h trend + volume confirmation
# - TK cross (Tenkan/Kijun) signals momentum shift
# - Trading in direction of cloud (price above/below) ensures trend alignment
# - 12h EMA50 trend filter ensures we trade with higher timeframe trend
# - Volume confirmation (2x average) reduces false signals
# - Works in bull (TK cross up in bullish cloud) and bear (TK cross down in bearish cloud)
# - Position size 0.25 targets ~50-100 trades over 4 years to avoid fee drag
# - Ichimoku is a proven institutional indicator that works across market regimes
# - Uses 12h for Ichimoku calculation and trend, 6h for execution timing
# - Similar Ichimoku strategies show promise in backtests when properly filtered