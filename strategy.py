#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_v1
Hypothesis: On 6h timeframe, Ichimoku Kumo twist (Senkou Span A/B cross) with 1d trend filter (price vs 1d EMA50) captures major trend reversals with low frequency. Works in both bull/bear markets by only taking trades in direction of higher timeframe trend. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1d data ONCE before loop for HTF trend filter and Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Kumo twist: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    senkou_span_a_shift = np.concatenate([[np.nan], senkou_span_a[:-1]])
    senkou_span_b_shift = np.concatenate([[np.nan], senkou_span_b[:-1]])
    
    bullish_twist = (senkou_span_a > senkou_span_b) & (senkou_span_a_shift <= senkou_span_b_shift)
    bearish_twist = (senkou_span_a < senkou_span_b) & (senkou_span_a_shift >= senkou_span_b_shift)
    
    # Align Ichimoku components and twist signals to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou Span B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(bullish_twist_aligned[i]) or
            np.isnan(bearish_twist_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Kumo twist signals
        twist_bullish = bullish_twist_aligned[i] > 0.5
        twist_bearish = bearish_twist_aligned[i] > 0.5
        
        # Price relative to Kumo (cloud)
        price_above_kumo = (close[i] > senkou_span_a_aligned[i]) & (close[i] > senkou_span_b_aligned[i])
        price_below_kumo = (close[i] < senkou_span_a_aligned[i]) & (close[i] < senkou_span_b_aligned[i])
        
        # Long logic: bullish Kumo twist in uptrend with price above cloud
        if twist_bullish and uptrend and price_above_kumo:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: bearish Kumo twist in downtrend with price below cloud
        elif twist_bearish and downtrend and price_below_kumo:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite twist or loss of trend
        elif position == 1 and (twist_bearish or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (twist_bullish or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_v1"
timeframe = "6h"
leverage = 1.0