#!/usr/bin/env python3
"""
6h_Ichimoku_Kijun_Tenkan_Cross_1wTrend_VolumeConfirm
Hypothesis: Uses 6h timeframe with Ichimoku TK cross filtered by 1w trend (price above/below weekly cloud) and volume confirmation. Works in bull/bear markets by only taking TK crosses aligned with the weekly trend. Weekly cloud acts as dynamic support/resistance. Target 12-30 trades/year (50-120 over 4 years) to minimize fee drag while capturing major trend shifts.
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
    
    # Get 1w data for weekly trend filter (cloud)
    df_1w = get_htf_data(prices, '1w')
    
    # Ichimoku weekly cloud: Senkou Span A and B
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over 9 periods
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over 26 periods
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over 52 periods shifted 26 periods ahead
    
    # Calculate Tenkan-sen (9-period) and Kijun-sen (26-period) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen: (9-period high + 9-period low)/2
    tenkan_period = 9
    max_high_9 = pd.Series(high_1w).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    min_low_9 = pd.Series(low_1w).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen: (26-period high + 26-period low)/2
    kijun_period = 26
    max_high_26 = pd.Series(high_1w).rolling(window=kijun_period, min_periods=kijun_period).max().values
    min_low_26 = pd.Series(low_1w).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A: (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B: (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_period = 52
    max_high_52 = pd.Series(high_1w).rolling(window=senkou_b_period, min_periods=senkou_b_period).max().values
    min_low_52 = pd.Series(low_1w).rolling(window=senkou_b_period, min_periods=senkou_b_period).min().values
    senkou_span_b = ((max_high_52 + min_low_52) / 2)
    
    # Shift Senkou Spans 26 periods ahead (they are plotted 26 periods into future)
    # For trend filter, we need current cloud, so we use unshifted values
    # The cloud at time t is formed by Senkou Span A and B from 26 periods ago
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    # First 26 values are invalid due to roll
    senkou_span_a_lagged[:26] = np.nan
    senkou_span_b_lagged[:26] = np.nan
    
    # Align weekly cloud and TK cross to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_lagged)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_lagged)
    
    # Weekly trend: price above cloud = bullish, below cloud = bearish
    # Cloud top = max(Senkou A, Senkou B), cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Get 1d data for volume confirmation (more stable than 6h volume)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    volume_confirm = vol_1d_aligned > (1.5 * vol_avg_1d_aligned)  # 1.5x average volume
    
    # TK Cross signals on 6h
    # Tenkan crossing above Kijun = bullish signal
    # Tenkan crossing below Kijun = bearish signal
    tk_cross_bull = (tenkan_sen_aligned > kijun_sen_aligned) & (np.roll(tenkan_sen_aligned, 1) <= np.roll(kijun_sen_aligned, 1))
    tk_cross_bear = (tenkan_sen_aligned < kijun_sen_aligned) & (np.roll(tenkan_sen_aligned, 1) >= np.roll(kijun_sen_aligned, 1))
    
    # Handle first value (no previous)
    tk_cross_bull[0] = False
    tk_cross_bear[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # Discrete size to minimize fee churn
    
    # Warmup: need weekly Ichimoku (52 period lookback + 26 shift) and 1d volume (20)
    start_idx = max(52 + 26 + 2*6, 20 + 2*6)  # ~156 bars for weekly Ichimoku warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(volume_confirm[i]) or np.isnan(tk_cross_bull[i]) or np.isnan(tk_cross_bear[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bull_cross = tk_cross_bull[i]
        bear_cross = tk_cross_bear[i]
        above_cloud = price_above_cloud[i]
        below_cloud = price_below_cloud[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: TK cross in direction of weekly trend with volume confirmation
            long_condition = bull_cross and above_cloud and vol_conf
            short_condition = bear_cross and below_cloud and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun (trend weakness) OR price breaks below cloud bottom
            exit_condition = (tk_cross_bear[i] or close_val < cloud_bottom[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun (trend weakness) OR price breaks above cloud top
            exit_condition = (tk_cross_bull[i] or close_val > cloud_top[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kijun_Tenkan_Cross_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0