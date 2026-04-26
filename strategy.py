#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm
Hypothesis: Ichimoku cloud breakout on 6h with 1d trend filter (price >/< Kumo twist) and volume confirmation (>1.5x average volume). 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
Designed to work in both bull and bear markets via 1d trend alignment and strict volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need warmup for Ichimoku (26*2)
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for breakout signals
    
    # Kumo (Cloud) twist: Senkou Span A > Senkou Span B = bullish cloud, < = bearish cloud
    # We'll use the 1d trend to filter: only take longs when 1d price > 1d Kumo, shorts when < 1d Kumo
    
    # Calculate 1d Ichimoku for trend filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # 1d Tenkan-sen (9-period)
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    # 1d Kijun-sen (26-period)
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    # 1d Senkou Span A
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    
    # 1d Senkou Span B (52-period)
    period52_high_1d = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = ((period52_high_1d + period52_low_1d) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 52 for Ichimoku, 20 for volume)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Current 6h Ichimoku values
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        span_a = senkou_span_a[i]
        span_b = senkou_span_b[i]
        
        # Current price
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # 1d trend filter values (aligned)
        tenkan_1d = tenkan_sen_1d_aligned[i]
        kijun_1d = kijun_sen_1d_aligned[i]
        span_a_1d = senkou_span_a_1d_aligned[i]
        span_b_1d = senkou_span_b_1d_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(span_a) or np.isnan(span_b) or
            np.isnan(tenkan_1d) or np.isnan(kijun_1d) or np.isnan(span_a_1d) or np.isnan(span_b_1d) or
            np.isnan(avg_vol)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Kumo (cloud) breakout conditions
        # Bullish breakout: price breaks above Kumo (both Span A and Span B) with Tenkan > Kijun
        bullish_kumo = span_a > span_b  # Bullish cloud
        price_above_kumo = close_val > span_a and close_val > span_b
        tenkan_above_kijun = tenkan > kijun
        
        # Bearish breakout: price breaks below Kumo (both Span A and Span B) with Tenkan < Kijun
        bearish_kumo = span_a < span_b  # Bearish cloud
        price_below_kumo = close_val < span_a and close_val < span_b
        tenkan_below_kijun = tenkan < kijun
        
        # 1d trend filter: only trade in direction of 1d trend
        # 1d bullish: price > 1d Kumo
        price_above_1d_kumo = close_val > span_a_1d and close_val > span_b_1d
        # 1d bearish: price < 1d Kumo
        price_below_1d_kumo = close_val < span_a_1d and close_val < span_b_1d
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: bullish Kumo breakout with 1d uptrend and volume confirmation
        long_condition = (bullish_kumo and price_above_kumo and tenkan_above_kijun and 
                         price_above_1d_kumo and volume_confirmed)
        # Short logic: bearish Kumo breakout with 1d downtrend and volume confirmation
        short_condition = (bearish_kumo and price_below_kumo and tenkan_below_kijun and 
                          price_below_1d_kumo and volume_confirmed)
        
        # Exit logic: price re-enters the Kumo (cloud)
        exit_long = close_val < span_a or close_val < span_b
        exit_short = close_val > span_a or close_val > span_b
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0