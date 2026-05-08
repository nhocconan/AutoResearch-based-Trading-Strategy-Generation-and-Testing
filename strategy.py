#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Ichimoku Cloud with Tenkan/Kijun Cross and Volume Confirmation
# Uses weekly Ichimoku cloud (from prior week) to identify support/resistance zones.
# Tenkan-sen/Kijun-sen cross provides entry signals, with price position relative to cloud filtering direction.
# In bullish weekly trend (price above cloud), look for long on Tenkan/Kijun cross up.
# In bearish weekly trend (price below cloud), look for short on Tenkan/Kijun cross down.
# Volume > 1.5x 20-period average confirms participation.
# Target: 10-25 trades/year (40-100 over 4 years) to minimize fee drag.

name = "6h_WeeklyIchimoku_Cross_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku cloud
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 52:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = np.full(len(weekly_high), np.nan)
    for i in range(8, len(weekly_high)):
        tenkan_sen[i] = (np.max(weekly_high[i-8:i+1]) + np.min(weekly_low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = np.full(len(weekly_high), np.nan)
    for i in range(25, len(weekly_high)):
        kijun_sen[i] = (np.max(weekly_high[i-25:i+1]) + np.min(weekly_low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = np.full(len(weekly_high), np.nan)
    for i in range(len(weekly_high)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = np.full(len(weekly_high), np.nan)
    for i in range(51, len(weekly_high)):
        senkou_span_b[i] = (np.max(weekly_high[i-51:i+1]) + np.min(weekly_low[i-51:i+1])) / 2
    
    # Get daily data for volume filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_volume = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(daily_volume), np.nan)
    if len(daily_volume) >= 20:
        for i in range(20, len(daily_volume)):
            vol_avg_20_daily[i] = np.mean(daily_volume[i-20:i])
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_weekly, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_weekly, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_b)
    
    # Align daily volume average to 6h timeframe
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            # Find current daily bar's volume
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                vol_daily_current = df_daily.iloc[idx_daily]['volume']
                vol_filter = vol_daily_current > 1.5 * vol_avg_20_daily_aligned[i]
        
        # Determine cloud boundaries and price position
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan/Kijun cross signals
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and \
                      tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and \
                        tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        if position == 0:
            # Look for entry: TK cross with price position and volume
            # Long when TK cross up and price above cloud (bullish alignment)
            long_condition = tk_cross_up and price_above_cloud and vol_filter
            
            # Short when TK cross down and price below cloud (bearish alignment)
            short_condition = tk_cross_down and price_below_cloud and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below cloud or TK cross down
            if close[i] < cloud_bottom or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above cloud or TK cross up
            if close[i] > cloud_top or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals