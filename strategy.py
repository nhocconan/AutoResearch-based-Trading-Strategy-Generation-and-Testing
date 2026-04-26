#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeFilter
Hypothesis: 6h Ichimoku cloud breakout (Tenkan-Kijun cross above/below cloud) in direction of 1w EMA50 trend, confirmed by volume spike (>2x 50-bar MA). Ichimoku provides dynamic support/resistance and trend direction. Weekly trend filter ensures alignment with higher timeframe momentum. Volume confirmation reduces false breakouts. Designed for 12-37 trades/year (50-150 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 1w trend while using Ichimoku cloud for precise entries.
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
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud top/bottom (Senkou Span A/B from 26 periods ago)
    # Shift forward by 26 to get current cloud (no look-ahead)
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    # First 26 values are invalid (rolled from end)
    senkou_a_current[:26] = np.nan
    senkou_b_current[:26] = np.nan
    
    # Cloud top (max of Senkou A/B) and bottom (min of Senkou A/B)
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # Volume confirmation: volume > 2x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (52 for Ichimoku, 50 for vol, 50 for 1w EMA)
    start_idx = max(52, 50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        vol_spike = volume_spike[i]
        
        # Determine 1w trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Ichimoku conditions:
        # Bullish: Tenkan > Kijun AND price above cloud
        # Bearish: Tenkan < Kijun AND price below cloud
        ichimoku_bullish = tenkan_val > kijun_val and close_val > cloud_top_val
        ichimoku_bearish = tenkan_val < kijun_val and close_val < cloud_bottom_val
        
        # Entry conditions: Ichimoku signal in trend direction with volume
        long_entry = ichimoku_bullish and bullish_1w and vol_spike
        short_entry = ichimoku_bearish and bearish_1w and vol_spike
        
        # Exit conditions: opposite Ichimoku signal or trend reversal
        exit_long = ichimoku_bearish or not bullish_1w
        exit_short = ichimoku_bullish or not bearish_1w
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0