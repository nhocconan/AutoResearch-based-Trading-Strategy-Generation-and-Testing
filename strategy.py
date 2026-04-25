#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudTwist_v1
Hypothesis: Use Ichimoku on 1d for trend direction and cloud twist (Senkou A/B cross) as regime filter, with TK cross on 6h for precise entry. 
In bull regime (price > cloud, Senkou A rising): long when TK crosses above Kijun. 
In bear regime (price < cloud, Senkou A falling): short when TK crosses below Kijun. 
Requires cloud twist confirmation (Senkou A/B cross in direction of trade) to avoid whipsaws. 
Position size: 0.25. Target: 50-150 total trades over 4 years.
Works in bull (trend-following with cloud support) and bear (trend-following with cloud resistance) markets.
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
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need sufficient data for Ichimoku
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
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = close_1d  # Not used for alignment as it's lagging
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate cloud twist: Senkou A/B cross
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_prev = np.roll(senkou_a_aligned, 1)
    senkou_b_prev = np.roll(senkou_b_aligned, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    bullish_twist = (senkou_a_aligned > senkou_b_aligned) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a_aligned < senkou_b_aligned) & (senkou_a_prev >= senkou_b_prev)
    
    # Regime filters
    price_above_cloud = close > np.maximum(senkou_a_aligned, senkou_b_aligned)
    price_below_cloud = close < np.minimum(senkou_a_aligned, senkou_b_aligned)
    senkou_a_rising = senkou_a_aligned > np.roll(senkou_a_aligned, 1)
    senkou_a_falling = senkou_a_aligned < np.roll(senkou_a_aligned, 1)
    
    # Bullish regime: price above cloud AND Senkou A rising
    bullish_regime = price_above_cloud & senkou_a_rising
    # Bearish regime: price below cloud AND Senkou A falling
    bearish_regime = price_below_cloud & senkou_a_falling
    
    # TK cross signals
    tenkan_prev = np.roll(tenkan_aligned, 1)
    kijun_prev = np.roll(kijun_aligned, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    tk_bullish_cross = (tenkan_aligned > kijun_aligned) & (tenkan_prev <= kijun_prev)
    tk_bearish_cross = (tenkan_aligned < kijun_aligned) & (tenkan_prev >= kijun_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52 periods)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long setup: bullish regime + TK bullish cross + bullish cloud twist
            long_setup = bullish_regime[i] and tk_bullish_cross[i] and bullish_twist[i]
            
            # Short setup: bearish regime + TK bearish cross + bearish cloud twist
            short_setup = bearish_regime[i] and tk_bearish_cross[i] and bearish_twist[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price crosses below Kijun OR regime turns bearish
            if (close[i] < kijun_aligned[i]) or (not bullish_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses above Kijun OR regime turns bullish
            if (close[i] > kijun_aligned[i]) or (bullish_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudTwist_v1"
timeframe = "6h"
leverage = 1.0