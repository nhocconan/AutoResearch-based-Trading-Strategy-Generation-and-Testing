#!/usr/bin/env python3
"""
6h_12h1d_Ichimoku_Breakout_v1
Hypothesis: Ichimoku TK cross with cloud filter from 12h/1d timeframes on 6h chart.
- Bullish: Tenkan > Kijun AND price > Senkou Span A/B (cloud) from 12h
- Bearish: Tenkan < Kijun AND price < Senkou Span A/B (cloud) from 1d
- Uses weekly trend filter (EMA50) to avoid counter-trend trades
- Target: 15-30 trades/year per symbol (60-120 over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Ichimoku cloud (primary trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # need 26*2 for Ichimoku
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku components (12h)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B: (52-period high + 52-period low) / 2 shifted 26 periods ahead
    high_52 = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b, additional_delay_bars=26)
    
    # Load 1d data for additional cloud confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (1d) for secondary confirmation
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((high_52_1d + low_52_1d) / 2)
    
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d, additional_delay_bars=26)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d, additional_delay_bars=26)
    
    # Load weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Cloud boundaries (top and bottom of cloud)
        cloud_top_12h = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom_12h = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_top_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross signals
        tk_bullish_12h = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish_12h = tenkan_aligned[i] < kijun_aligned[i]
        tk_bullish_1d = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_bearish_1d = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Price relative to cloud
        price_above_cloud_12h = price > cloud_top_12h
        price_below_cloud_12h = price < cloud_bottom_12h
        price_above_cloud_1d = price > cloud_top_1d
        price_below_cloud_1d = price < cloud_bottom_1d
        
        # Weekly trend filter
        weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] if i > 0 else True
        weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: bullish TK cross + price above cloud (both timeframes) + weekly uptrend
            if (tk_bullish_12h and tk_bullish_1d and 
                price_above_cloud_12h and price_above_cloud_1d and 
                weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below cloud (both timeframes) + weekly downtrend
            elif (tk_bearish_12h and tk_bearish_1d and 
                  price_below_cloud_12h and price_below_cloud_1d and 
                  weekly_downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below cloud OR TK cross turns bearish
            if (price < cloud_bottom_12h or tk_bearish_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud OR TK cross turns bullish
            if (price > cloud_top_12h or tk_bullish_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h1d_Ichimoku_Breakout_v1"
timeframe = "6h"
leverage = 1.0