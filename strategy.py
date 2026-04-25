#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Ichimoku cloud (from 1d) acts as dynamic support/resistance with TK cross for momentum. Weekly pivot provides directional bias (above/below weekly pivot = bull/bear). Volume confirmation ensures participation. Works in bull (long when price > cloud + TK cross bullish + price > weekly pivot) and bear (short when price < cloud + TK cross bearish + price < weekly pivot). Targets 50-150 trades over 4 years on 6h timeframe.
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
    
    # Get 1d data for Ichimoku cloud (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 1w data for weekly pivot (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku (52 for Senkou Span B) and volume MA
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # TK cross: Tenkan-sen > Kijun-sen = bullish, Tenkan-sen < Kijun-sen = bearish
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun
        
        # Price relative to cloud: above cloud = bullish, below cloud = bearish, inside cloud = neutral
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Weekly pivot bias: above pivot = bullish bias, below pivot = bearish bias
        price_above_weekly_pivot = curr_close > weekly_pivot_val
        price_below_weekly_pivot = curr_close < weekly_pivot_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price above cloud, TK cross bullish, price above weekly pivot, volume confirmation
            long_entry = price_above_cloud and tk_bullish and price_above_weekly_pivot and volume_confirm
            # Short: price below cloud, TK cross bearish, price below weekly pivot, volume confirmation
            short_entry = price_below_cloud and tk_bearish and price_below_weekly_pivot and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below cloud OR TK cross turns bearish
            if curr_close < cloud_bottom or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above cloud OR TK cross turns bullish
            if curr_close > cloud_top or tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0