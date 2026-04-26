#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter
Hypothesis: Use Ichimoku cloud from 6h for trend direction and support/resistance, with weekly trend filter (price above/below weekly EMA20) to avoid counter-trend trades. Enter on TK cross (Tenkan/Kijun) when price breaks cloud in direction of weekly trend. Volume confirmation (>1.5x 24-period average) adds conviction. Discrete sizing 0.25. Designed to capture strong trending moves while avoiding whipsaws in ranging markets. Weekly filter adapts to bull/bear regimes.
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
    open_time = prices['open_time'].values
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need enough for Senkou Span B
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Ichimoku components (6h)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h (they are already 6h, but need alignment for safety)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Weekly trend filter: EMA20 on weekly close
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume average (24-period, ~4 days on 6h) for confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku (52), weekly EMA (20), volume MA (24)
    start_idx = max(52, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
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
        ema_20_1w_val = ema_20_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Volume confirmation: current volume > 1.5x 24-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        # TK cross signals
        tk_cross_up = tenkan_val > kijun_val
        tk_cross_down = tenkan_val < kijun_val
        
        if position == 0:
            # Long: price above cloud, TK cross up, weekly uptrend (price > weekly EMA20), volume confirmation
            long_signal = (close_val > cloud_top) and tk_cross_up and (close_val > ema_20_1w_val) and volume_confirmed
            # Short: price below cloud, TK cross down, weekly downtrend (price < weekly EMA20), volume confirmation
            short_signal = (close_val < cloud_bottom) and tk_cross_down and (close_val < ema_20_1w_val) and volume_confirmed
            
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
            # Exit: price closes below cloud bottom OR TK cross down
            if (close_val < cloud_bottom) or tk_cross_down:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above cloud top OR TK cross up
            if (close_val > cloud_top) or tk_cross_up:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0