#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Filter
Hypothesis: Uses 6h Ichimoku cloud twist (Tenkan/Kijun cross) filtered by 1d EMA50 trend and volume spike confirmation. 
The Kumo twist signals potential trend changes, while the 1d EMA50 ensures we only trade in the direction of the higher timeframe trend. 
Volume confirmation reduces false signals. Designed for low frequency (target 12-30 trades/year) to work in both bull and bear markets.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (cloud) twist: Tenkan crosses Kijun
    # Bullish twist: Tenkan crosses above Kijun
    # Bearish twist: Tenkan crosses below Kijun
    tenkan_prev = np.roll(tenkan, 1)
    kijun_prev = np.roll(kijun, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    bullish_twist = (tenkan > kijun) & (tenkan_prev <= kijun_prev)
    bearish_twist = (tenkan < kijun) & (tenkan_prev >= kijun_prev)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # Discrete size to minimize fee churn
    
    # Warmup: need Ichimoku (52), 1d EMA50 (50), vol avg (20)
    start_idx = max(52, 50 + 6*4, 20)  # 52 bars for Ichimoku, plus 1d EMA50 warmup (4 6h bars per day)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        bull_twist = bullish_twist[i]
        bear_twist = bearish_twist[i]
        
        if position == 0:
            # Look for entry: Kumo twist with 1d EMA50 alignment and volume confirmation
            long_condition = bull_twist and (close_val > ema_val) and vol_conf
            short_condition = bear_twist and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun (reverse twist) OR price closes below 1d EMA50
            if bearish_twist[i] or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun (reverse twist) OR price closes above 1d EMA50
            if bullish_twist[i] or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0