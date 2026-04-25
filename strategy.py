#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrendFilter
Hypothesis: Ichimoku Kumo (cloud) twist combined with 1d trend filter captures trend changes with low frequency.
Kumo twist (Senkou Span A/B cross) signals potential trend reversal. 1d EMA50 filter ensures alignment with higher timeframe trend.
Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years) with discrete position sizing (0.25) to minimize fee drag.
Works in bull markets via trend continuation signals and bear markets via trend reversal signals from Kumo twist.
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
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter (loaded ONCE)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Kumo twist detection: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_shift = np.roll(senkou_a, 1)
    senkou_b_shift = np.roll(senkou_b, 1)
    senkou_a_shift[0] = np.nan
    senkou_b_shift[0] = np.nan
    
    bullish_twist = (senkou_a > senkou_b) & (senkou_a_shift <= senkou_b_shift)
    bearish_twist = (senkou_a < senkou_b) & (senkou_a_shift >= senkou_b_shift)
    
    # Align Ichimoku and twist signals to 6h timeframe (completed candle)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)  # Using 1d as reference for alignment
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) + 1d EMA (50)
    start_idx = max(52, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or np.isnan(bullish_twist_aligned[i]) or 
            np.isnan(bearish_twist_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Kumo twist + 1d EMA50 trend alignment
            long_entry = bullish_twist_aligned[i] == 1.0 and curr_close > ema_50_1d_aligned[i]
            short_entry = bearish_twist_aligned[i] == 1.0 and curr_close < ema_50_1d_aligned[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when bearish twist occurs or price closes below Kijun-sen
            if bearish_twist_aligned[i] == 1.0 or curr_close < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when bullish twist occurs or price closes above Kijun-sen
            if bullish_twist_aligned[i] == 1.0 or curr_close > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrendFilter"
timeframe = "6h"
leverage = 1.0