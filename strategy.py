#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v4
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with weekly trend filter and daily cloud confirmation. 
TK cross provides timely momentum signals, weekly trend ensures alignment with higher timeframe direction, 
and daily cloud acts as a dynamic support/resistance filter. This combination should work in both bull 
and bear markets by only taking trades in the direction of the weekly trend when price is above/below 
the daily cloud. Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to 
minimize fee drag. Uses proper MTF alignment via mtf_data helpers.
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Load daily data ONCE before loop for cloud and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA50 for trend filter (close > EMA50 = uptrend)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_trend = np.where(ema_50_1w_aligned > 0, 
                            np.where(close > ema_50_1w_aligned, 1, -1), 
                            0)
    
    # Calculate daily Ichimoku components for cloud
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 6h Tenkan and Kijun for TK cross
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2
    
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2
    
    # TK cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Price relative to daily cloud
    # Price above cloud: close > max(senkou_a, senkou_b)
    # Price below cloud: close < min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 26 for 6h Kijun)
    start_idx = max(52, 26, 26)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry logic
        if position == 0:
            # Long: TK cross up AND weekly uptrend AND price above daily cloud
            if tk_cross_up[i] and weekly_trend[i] == 1 and price_above_cloud[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down AND weekly downtrend AND price below daily cloud
            elif tk_cross_down[i] and weekly_trend[i] == -1 and price_below_cloud[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down OR weekly trend turns down OR price falls below cloud
            if tk_cross_down[i] or weekly_trend[i] == -1 or close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR weekly trend turns up OR price rises above cloud
            if tk_cross_up[i] or weekly_trend[i] == 1 or close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v4"
timeframe = "6h"
leverage = 1.0