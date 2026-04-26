#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_WeeklyTrend_Filter_v3
Hypothesis: 6h Ichimoku Cloud (TK Cross + price vs cloud) with 1w trend filter (price above/below weekly EMA50) and volume spike confirmation. 
Designed to capture medium-term trends with low trade frequency (12-30/year) by requiring alignment across 6h, 1d, and 1w timeframes. 
Works in bull markets (price above cloud + weekly uptrend) and bear markets (price below cloud + weekly downtrend). 
Uses discrete position sizing (0.25) to minimize fee churn. 
Ichimoku provides dynamic support/resistance via cloud, reducing whipsaws in ranging markets.
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
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Ichimoku components (1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Weekly trend filter: EMA50 on 1w
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 1.8x average volume (moderate to balance frequency and reliability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku (52), volume MA (20), weekly EMA (50)
    start_idx = max(52, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + weekly uptrend + volume spike
            tk_bullish = tenkan_val > kijun_val
            price_above_cloud = close_val > cloud_top
            weekly_uptrend = close_val > ema_50_1w_val
            volume_spike = volume_val > 1.8 * vol_ma_val
            
            if tk_bullish and price_above_cloud and weekly_uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: TK cross bearish + price below cloud + weekly downtrend + volume spike
            elif (tenkan_val < kijun_val and 
                  close_val < cloud_bottom and 
                  close_val < ema_50_1w_val and 
                  volume_val > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross bearish or price drops below cloud
            if (tenkan_val < kijun_val) or (close_val < cloud_top):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross bullish or price rises above cloud
            if (tenkan_val > kijun_val) or (close_val > cloud_bottom):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyTrend_Filter_v3"
timeframe = "6h"
leverage = 1.0