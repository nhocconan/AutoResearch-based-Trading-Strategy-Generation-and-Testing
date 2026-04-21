#!/usr/bin/env python3
"""
6h_1d_1w_Ichimoku_Cloud_Breakout_v1
Hypothesis: Ichimoku cloud breakout on 6h with 1d trend filter (price > Kumo) and 1w bias (Senkou Span A > B).
Works in bull/bear: In uptrend (1w bullish), long when price breaks above Kumo; in downtrend (1w bearish), short when price breaks below Kumo.
Uses volume confirmation to avoid false breakouts. Target: 12-25 trades/year per symbol (50-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku calculation (needs 26 periods)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
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
    
    # Align Ichimoku components to 6h timeframe (default delay=1 for completed bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo (cloud) boundaries: max/min of Senkou Span A and B
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # 1w data for weekly bias (Senkou Span A > B = bullish bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Ichimoku for bias (same calculation)
    period26_high_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    period52_high_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    
    tenkan_sen_1w = (period26_high_1w + period26_low_1w) / 2
    kijun_sen_1w = (period26_high_1w + period26_low_1w) / 2
    senkou_span_a_1w = (tenkan_sen_1w + kijun_sen_1w) / 2
    senkou_span_b_1w = (period52_high_1w + period52_low_1w) / 2
    
    # Weekly bullish bias: Senkou Span A > Senkou Span B
    weekly_bullish_bias = senkou_span_a_1w > senkou_span_b_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish_bias.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Determine weekly bias (convert aligned float back to boolean)
        weekly_bullish = weekly_bullish_aligned[i] > 0.5 if not np.isnan(weekly_bullish_aligned[i]) else True
        
        if position == 0:
            # Long conditions: price breaks above Kumo TOP AND 1d bullish (price > Kumo) AND weekly bullish bias AND volume
            if (price > kumo_top[i] and 
                price > ((kumo_top[i] + kumo_bottom[i]) / 2) and  # price above cloud midpoint
                weekly_bullish and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Kumo BOTTOM AND 1d bearish (price < Kumo) AND weekly bearish bias AND volume
            elif (price < kumo_bottom[i] and 
                  price < ((kumo_top[i] + kumo_bottom[i]) / 2) and  # price below cloud midpoint
                  not weekly_bullish and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below Kumo BOTTOM (cloud break) or Tenkan-sen < Kijun-sen (momentum loss)
            if price < kumo_bottom[i] or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above Kumo TOP (cloud break) or Tenkan-sen > Kijun-sen (momentum loss)
            if price > kumo_top[i] or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0