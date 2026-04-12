#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_ichimoku_cloud_trend_v1
# Uses Ichimoku cloud from daily timeframe as primary trend filter (Kijun-sen and Senkou Span A/B).
# Entry signals triggered when Tenkan-sen crosses Kijun-sen on 6h chart with cloud confirmation.
# Uses weekly Ichimoku for higher timeframe trend bias (avoid counter-trend trades).
# Designed for low trade frequency with clear trend-following logic.
# Works in both bull and bear markets by following Ichimoku trend signals.

name = "6h_1d_1w_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for entry, but available for confirmation if needed
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate weekly Ichimoku for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Kijun-sen for trend filter
    period26_high_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen_1w = (period26_high_1w + period26_low_1w) / 2
    
    # Align weekly Kijun-sen to 6h timeframe
    kijun_sen_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen_1w)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(kijun_sen_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine Ichimoku cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Weekly trend filter: price above/below weekly Kijun-sen
        weekly_uptrend = close[i] > kijun_sen_1w_aligned[i]
        weekly_downtrend = close[i] < kijun_sen_1w_aligned[i]
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Tenkan-sen/Kijun-sen crossover signals
        tenkan_prev = tenkan_sen_aligned[i-1] if i > 0 else tenkan_sen_aligned[i]
        kijun_prev = kijun_sen_aligned[i-1] if i > 0 else kijun_sen_aligned[i]
        tenkan_curr = tenkan_sen_aligned[i]
        kijun_curr = kijun_sen_aligned[i]
        
        # Bullish crossover: Tenkan crosses above Kijun
        bullish_cross = (tenkan_prev <= kijun_prev) and (tenkan_curr > kijun_curr)
        # Bearish crossover: Tenkan crosses below Kijun
        bearish_cross = (tenkan_prev >= kijun_prev) and (tenkan_curr < kijun_curr)
        
        # Long signal: bullish crossover above cloud with weekly uptrend
        if bullish_cross and tenkan_curr > cloud_top and weekly_uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: bearish crossover below cloud with weekly downtrend
        elif bearish_cross and tenkan_curr < cloud_bottom and weekly_downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit when Tenkan-sen crosses back through Kijun-sen (opposite signal)
        elif position == 1 and bearish_cross:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_cross:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals