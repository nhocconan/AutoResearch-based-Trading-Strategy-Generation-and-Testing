#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) on 6h with price above/below daily cloud acts as trend continuation filter. Works in bull/bear by aligning with higher timeframe cloud color (green=bull, red=bear). Targets 12-30 trades/year with discrete sizing (0.25) to avoid fee drag.
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
    
    # Load 6h data for Ichimoku calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    # Load 1d data for trend/cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Ichimoku calculations on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_6h['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_6h['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_6h['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_6h['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_6h['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_6h['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h (wait for completed 6h bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Daily cloud for trend filter
    # Daily Tenkan-sen (9-period)
    daily_tenkan_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    daily_tenkan_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    daily_tenkan = (daily_tenkan_high + daily_tenkan_low) / 2
    
    # Daily Kijun-sen (26-period)
    daily_kijun_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    daily_kijun_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    daily_kijun = (daily_kijun_high + daily_kijun_low) / 2
    
    # Daily Senkou Span A
    daily_senkou_a = ((daily_tenkan + daily_kijun) / 2)
    
    # Daily Senkou Span B (52-period)
    daily_period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    daily_period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    daily_senkou_b = ((daily_period52_high + daily_period52_low) / 2)
    
    # Align daily cloud to 6h
    daily_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, daily_senkou_a)
    daily_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, daily_senkou_b)
    
    # Volume confirmation: 2x average volume
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 * 6h = 4d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Ichimoku calculations (52) and volume (24)
    start_idx = max(52, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        daily_senkou_a_val = daily_senkou_a_aligned[i]
        daily_senkou_b_val = daily_senkou_b_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(senkou_a_val) or 
            np.isnan(senkou_b_val) or np.isnan(daily_senkou_a_val) or 
            np.isnan(daily_senkou_b_val) or np.isnan(avg_vol)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation
        volume_confirmed = vol > 2.0 * avg_vol
        
        # TK Cross
        tk_cross_up = tenkan_val > kijun_val
        tk_cross_down = tenkan_val < kijun_val
        
        # Price relative to 6h cloud
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        price_above_6h_cloud = close_val > cloud_top
        price_below_6h_cloud = close_val < cloud_bottom
        
        # Daily cloud color (trend filter)
        daily_cloud_top = max(daily_senkou_a_val, daily_senkou_b_val)
        daily_cloud_bottom = min(daily_senkou_a_val, daily_senkou_b_val)
        daily_cloud_green = daily_senkou_a_val > daily_senkou_b_val  # bullish cloud
        daily_cloud_red = daily_senkou_a_val < daily_senkou_b_val    # bearish cloud
        
        # Long: TK cross up + price above 6h cloud + daily cloud green + volume
        long_condition = tk_cross_up and price_above_6h_cloud and daily_cloud_green and volume_confirmed
        # Short: TK cross down + price below 6h cloud + daily cloud red + volume
        short_condition = tk_cross_down and price_below_6h_cloud and daily_cloud_red and volume_confirmed
        
        # Exit: TK cross in opposite direction
        long_exit = position == 1 and tk_cross_down
        short_exit = position == -1 and tk_cross_up
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0