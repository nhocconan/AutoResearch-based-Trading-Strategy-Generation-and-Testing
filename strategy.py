#!/usr/bin/env python3
name = "6h_Ichimoku_TenkanKijun_Cross_1dCloud_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for Ichimoku components and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1d EMA(50) for additional trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Wait for Senkou Span B
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud (bullish), and price above EMA(50)
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i] and 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud (bearish), and price below EMA(50)
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i] and 
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Tenkan crosses below Kijun or price below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] or 
                close[i] < senkou_a_aligned[i] or close[i] < senkou_b_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses above Kijun or price above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] or 
                close[i] > senkou_a_aligned[i] or close[i] > senkou_b_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter and EMA(50) trend filter.
# Ichimoku provides a comprehensive trend system: Tenkan-Kijun cross signals momentum shifts,
# the cloud (Senkou A/B) acts as dynamic support/resistance, and EMA(50) confirms the higher timeframe trend.
# In bullish regimes, we buy when Tenkan crosses above Kijun with price above the cloud.
# In bearish regimes, we sell when Tenkan crosses below Kijun with price below the cloud.
# The cloud filter ensures we only trade in the direction of the higher timeframe trend,
# reducing whipsaws in sideways markets. Position size 0.25 balances risk and reward.
# This strategy works in both bull markets (trend following) and bear markets (counter-trend reversals at cloud edges).