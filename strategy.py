#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Ichimoku cloud (TK cross + price above/below cloud) with 1w trend filter (price vs Kumo twist) and volume confirmation captures strong momentum swings. Works in bull/bear by only taking trades aligned with weekly Ichimoku trend. Targets 12-25 trades/year via strict entry conditions. Uses discrete sizing (0.25) to control drawdown.
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Weekly Ichimoku components for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Conversion line (9-period): (highest high + lowest low)/2 over 9 periods
    conversion_line_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                         pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    # Base line (26-period): (highest high + lowest low)/2 over 26 periods
    base_line_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                   pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    # Leading Span A: (Conversion line + Base line)/2
    leading_span_a_1w = (conversion_line_1w + base_line_1w) / 2
    # Leading Span B: (highest high + lowest low)/2 over 52 periods
    leading_span_b_1w = (pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                        pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2
    
    # Kumo twist: Leading Span A > Leading Span B = bullish cloud, < = bearish
    kumo_twist_bullish = leading_span_a_1w > leading_span_b_1w
    kumo_twist_bearish = leading_span_a_1w < leading_span_b_1w
    
    # Align weekly trend to 6h
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1w, kumo_twist_bullish.values.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1w, kumo_twist_bearish.values.astype(float))
    
    # 6h Ichimoku components for entry signals
    # Conversion line (9-period)
    conversion_line = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                      pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Base line (26-period)
    base_line = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # Leading Span A
    leading_span_a = (conversion_line + base_line) / 2
    # Leading Span B (52-period)
    leading_span_b = (pd.Series(high).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low).rolling(window=52, min_periods=52).min()) / 2
    
    # TK cross: Conversion line crosses Base line
    tk_cross_bullish = conversion_line > base_line
    tk_cross_bearish = conversion_line < base_line
    
    # Price relative to cloud: price > max(Span A, Span B) = above cloud
    # price < min(Span A, Span B) = below cloud
    cloud_top = np.maximum(leading_span_a, leading_span_b)
    cloud_bottom = np.minimum(leading_span_a, leading_span_b)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of all indicators (52 for weekly, 52 for 6h Span B)
    start_idx = max(52, 52)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumo_twist_bullish_aligned[i]) or np.isnan(kumo_twist_bearish_aligned[i]) or
            np.isnan(conversion_line[i]) or np.isnan(base_line[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Weekly trend filter from Ichimoku cloud twist
        weekly_bullish = bool(kumo_twist_bullish_aligned[i])
        weekly_bearish = bool(kumo_twist_bearish_aligned[i])
        
        # 6h Ichimoku entry conditions
        tk_bullish = tk_cross_bullish[i]
        tk_bearish = tk_cross_bearish[i]
        price_above = price_above_cloud[i]
        price_below = price_below_cloud[i]
        vol_conf = volume_confirm[i]
        
        # Long: TK bullish cross + price above cloud + weekly bullish + volume
        long_entry = tk_bullish and price_above and weekly_bullish and vol_conf
        # Short: TK bearish cross + price below cloud + weekly bearish + volume
        short_entry = tk_bearish and price_below and weekly_bearish and vol_conf
        
        # Exit: opposite TK cross or price re-enters cloud
        long_exit = tk_bearish or (close[i] < cloud_top[i] and close[i] > cloud_bottom[i])
        short_exit = tk_bullish or (close[i] > cloud_bottom[i] and close[i] < cloud_top[i])
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0