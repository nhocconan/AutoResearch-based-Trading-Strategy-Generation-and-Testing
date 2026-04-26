#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Twist_WeeklyTrendFilter_VolumeConfirm_v1
Hypothesis: Ichimoku TK cross with cloud twist (price above/below cloud) on 6h, filtered by weekly trend (price > weekly EMA50 = uptrend, < = downtrend) and volume confirmation (1.5x average). Weekly trend provides robust regime filter for BTC/ETH in both bull/bear markets, reducing false signals during sideways periods. Targets 50-150 total trades over 4 years.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 6h data for Ichimoku components
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align weekly trend to 6h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align Ichimoku components to 6h (they are already on 6h, but need to shift for cloud)
    # Senkou Span A and B are plotted 26 periods ahead, so we need to shift them back for current cloud
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    # First 26 values are invalid due to roll
    senkou_span_a_lagged[:26] = np.nan
    senkou_span_b_lagged[:26] = np.nan
    
    # Alink Ichimoku to LTF (6h to 6h is identity, but we use align for consistency)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a_lagged)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b_lagged)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly EMA50 (50), Ichimoku (52), volume MA (20)
    start_idx = max(50, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        tenkan_sen_val = tenkan_sen_aligned[i]
        kijun_sen_val = kijun_sen_aligned[i]
        senkou_span_a_val = senkou_span_a_aligned[i]
        senkou_span_b_val = senkou_span_b_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_span_a_val, senkou_span_b_val)
        cloud_bottom = min(senkou_span_a_val, senkou_span_b_val)
        
        # TK cross
        tk_cross_bull = tenkan_sen_val > kijun_sen_val
        tk_cross_bear = tenkan_sen_val < kijun_sen_val
        
        # Price relative to cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        if position == 0:
            # Long: bullish TK cross, price above cloud, weekly uptrend, volume confirmation
            long_signal = tk_cross_bull and price_above_cloud and (close_val > ema_50_1w_val) and (volume_val > 1.5 * vol_ma_val)
            # Short: bearish TK cross, price below cloud, weekly downtrend, volume confirmation
            short_signal = tk_cross_bear and price_below_cloud and (close_val < ema_50_1w_val) and (volume_val > 1.5 * vol_ma_val)
            
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
            # Exit: bearish TK cross OR price falls below cloud OR weekly trend turns down
            if tk_cross_bear or (close_val < cloud_bottom) or (close_val < ema_50_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish TK cross OR price rises above cloud OR weekly trend turns up
            if tk_cross_bull or (close_val > cloud_top) or (close_val > ema_50_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Twist_WeeklyTrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0