#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Trend_1wBias"
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
    
    # Load weekly data ONCE for bias filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Load daily data ONCE for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily data
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan = (high_1d.rolling(window=9, min_periods=9).max() + 
              low_1d.rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun = (high_1d.rolling(window=26, min_periods=26).max() + 
             low_1d.rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_b = ((high_1d.rolling(window=52, min_periods=52).max() + 
                 low_1d.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Convert to numpy arrays
    tenkan_arr = tenkan.values
    kijun_arr = kijun.values
    senkou_a_arr = senkou_a.values
    senkou_b_arr = senkou_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_arr)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_arr)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_arr)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_arr)
    
    # Weekly bias: price above/below weekly cloud
    weekly_high = df_1w['high']
    weekly_low = df_1w['low']
    weekly_close = df_1w['close']
    
    # Weekly Tenkan and Kijun
    wk_tenkan = (weekly_high.rolling(window=9, min_periods=9).max() + 
                 weekly_low.rolling(window=9, min_periods=9).min()) / 2
    wk_kijun = (weekly_high.rolling(window=26, min_periods=26).max() + 
                weekly_low.rolling(window=26, min_periods=26).min()) / 2
    
    # Weekly Senkou Span A and B
    wk_senkou_a = ((wk_tenkan + wk_kijun) / 2).shift(26)
    wk_senkou_b = ((weekly_high.rolling(window=52, min_periods=52).max() + 
                    weekly_low.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align weekly cloud to 6h
    wk_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_a.values)
    wk_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_b.values)
    
    # Weekly cloud boundaries (future cloud)
    wk_cloud_top = np.maximum(wk_senkou_a_aligned, wk_senkou_b_aligned)
    wk_cloud_bottom = np.minimum(wk_senkou_a_aligned, wk_senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52 + 26  # Wait for Ichimoku to be valid
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(wk_cloud_top[i]) or np.isnan(wk_cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current Ichimoku values
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Current cloud boundaries (leading spans)
        cloud_top = np.maximum(senkou_a_val, senkou_b_val)
        cloud_bottom = np.minimum(senkou_a_val, senkou_b_val)
        
        # Weekly bias
        price_above_wk_cloud = close[i] > wk_cloud_top[i]
        price_below_wk_cloud = close[i] < wk_cloud_bottom[i]
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, weekly bullish bias
            tk_bullish = tenkan_val > kijun_val
            price_above_cloud = close[i] > cloud_top
            
            if tk_bullish and price_above_cloud and price_above_wk_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, weekly bearish bias
            elif not tk_bullish and close[i] < cloud_bottom and price_below_wk_cloud:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bearish or price drops below cloud
            if tenkan_aligned[i] <= kijun_aligned[i] or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish or price rises above cloud
            if tenkan_aligned[i] >= kijun_aligned[i] or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku cloud system on 6h with weekly bias filter
# - Uses Ichimoku (Tenkan/Kijun cross + cloud) on daily timeframe aligned to 6h
# - Weekly Ichimoku cloud acts as bias filter: only take longs when price above weekly cloud
# - Weekly cloud acts as bias filter: only take shorts when price below weekly cloud
# - Works in bull markets: TK bullish cross + price above daily cloud + price above weekly cloud
# - Works in bear markets: TK bearish cross + price below daily cloud + price below weekly cloud
# - Cloud acts as dynamic support/resistance reducing whipsaws
# - Position size 0.25 balances return vs fee drag (target ~30-80 trades/year)
# - Ichimoku is proven effective in crypto (ranked Tier 8 in program.md)
# - Weekly bias filter prevents counter-trend trades in strong weekly trends
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Uses actual Ichimoku formulas, not approximations
# - Novel combination: Daily Ichimoku signals + Weekly Ichimoku bias filter (not recently tried)