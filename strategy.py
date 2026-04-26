#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_12hTrend_Confirmation
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 12h cloud filter for trend direction and weekly ADX for regime.
Ichimoku provides objective trend/momentum signals while higher timeframes filter for confluence.
Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25).
Works in both bull and bear markets by requiring alignment with 12h cloud and weekly ADX > 20.
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_12h['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_12h['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_12h['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_12h['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_12h['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_12h['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Load weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX (14-period)
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # ADX
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align weekly ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(52, 26, 14) + 30  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Determine cloud color and price position relative to cloud
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Long logic: TK cross bullish + price above cloud + weekly ADX > 20 (trending market)
        if (tenkan_aligned[i] > kijun_aligned[i] and 
            tenkan_aligned[i-1] <= kijun_aligned[i-1] and  # Fresh cross
            price_above_cloud and 
            adx_aligned[i] > 20):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK cross bearish + price below cloud + weekly ADX > 20 (trending market)
        elif (tenkan_aligned[i] < kijun_aligned[i] and 
              tenkan_aligned[i-1] >= kijun_aligned[i-1] and  # Fresh cross
              price_below_cloud and 
              adx_aligned[i] > 20):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: TK cross in opposite direction OR price enters cloud
        elif position == 1 and (tenkan_aligned[i] < kijun_aligned[i] or not price_above_cloud):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tenkan_aligned[i] > kijun_aligned[i] or not price_below_cloud):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_12hTrend_Confirmation"
timeframe = "6h"
leverage = 1.0