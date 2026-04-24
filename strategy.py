#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud strategy with 12h trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for trend filter (price above/below Kumo cloud) and TK cross timing.
- Ichimoku components from 6h: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52 displaced).
- Entry: Long when TK cross bullish AND price > Kumo (bullish cloud) AND 12h trend up AND volume > 1.5 * 6h volume MA(20);
         Short when TK cross bearish AND price < Kumo (bearish cloud) AND 12h trend down AND volume > 1.5 * 6h volume MA(20).
- Exit: Opposite TK cross or price crosses Kumo in opposite direction.
- Signal size: 0.25 discrete to control fee drag.
- Designed to capture trends with Ichimoku's built-in support/resistance (Kumo) and momentum (TK cross).
- Works in bull markets via longs in bullish cloud, bear markets via shorts in bearish cloud.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to primary 6h timeframe (no additional delay needed as they're based on completed 6h bars)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Get 12h data for trend filter (price vs Kumo)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Ichimoku on 12h for trend filter
    period9_high_12h = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low_12h = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_12h = (period9_high_12h + period9_low_12h) / 2
    
    period26_high_12h = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low_12h = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_12h = (period26_high_12h + period26_low_12h) / 2
    
    senkou_a_12h = (tenkan_12h + kijun_12h) / 2
    
    period52_high_12h = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low_12h = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b_12h = (period52_high_12h + period52_low_12h) / 2
    
    # Align 12h Ichimoku components
    span_a_12h_aligned = align_htf_to_ltf(prices, df_12h, senkou_a_12h)
    span_b_12h_aligned = align_htf_to_ltf(prices, df_12h, senkou_b_12h)
    
    # Calculate volume MA(20) for 6h timeframe
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max of 52 for Ichimoku, 20 for volume)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(span_a_12h_aligned[i]) or np.isnan(span_b_12h_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Kumo (cloud) boundaries: Senkou Span A and B
        # Bullish cloud: Span A > Span B, Bearish cloud: Span A < Span B
        span_a_6h = span_a_aligned[i]
        span_b_6h = span_b_aligned[i]
        span_a_12h = span_a_12h_aligned[i]
        span_b_12h = span_b_12h_aligned[i]
        
        # Kumo trend: price relative to cloud
        bullish_cloud_6h = span_a_6h > span_b_6h
        bearish_cloud_6h = span_a_6h < span_b_6h
        bullish_cloud_12h = span_a_12h > span_b_12h
        bearish_cloud_12h = span_a_12h < span_b_12h
        
        # Price above/below cloud
        price_above_kumo = curr_close > max(span_a_6h, span_b_6h)
        price_below_kumo = curr_close < min(span_a_6h, span_b_6h)
        
        # TK cross
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Bullish TK cross AND price above Kumo AND bullish 12h trend
                if tk_bullish and price_above_kumo and bullish_cloud_12h:
                    signals[i] = 0.25
                    position = 1
                # Short: Bearish TK cross AND price below Kumo AND bearish 12h trend
                elif tk_bearish and price_below_kumo and bearish_cloud_12h:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit on bearish TK cross OR price drops below Kumo
            if not tk_bullish or not price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on bullish TK cross OR price rises above Kumo
            if not tk_bearish or not price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0