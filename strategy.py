#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_TK_Cross_v1
Hypothesis: 6h Ichimoku system with Kumo twist (Senkou Span A/B cross) as trend filter and TK line cross for entry. Works in bull/bear markets: Kumo twist identifies major trend changes, TK cross provides timely entries with trend alignment. Uses 12h/1d HTF for multi-timeframe confirmation to reduce whipsaw. Targets 12-25 trades/year by requiring Kumo twist confirmation and volume filter. Designed to capture strong trends while avoiding sideways chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Calculate Kumo twist (Senkou Span A/B cross) - trend change signal
    # Kumo twist bullish: Senkou Span A crosses above Senkou Span B
    # Kumo twist bearish: Senkou Span A crosses below Senkou Span B
    senkou_span_a_prev = senkou_span_a.shift(1)
    senkou_span_b_prev = senkou_span_b.shift(1)
    
    kumo_twist_bullish = (senkou_span_a > senkou_span_b) & (senkou_span_a_prev <= senkou_span_b_prev)
    kumo_twist_bearish = (senkou_span_a < senkou_span_b) & (senkou_span_a_prev >= senkou_span_b_prev)
    
    # Calculate Kumo twist strength (distance between spans)
    kumo_thickness = np.abs(senkou_span_a - senkou_span_b)
    kumo_thickness_ma = pd.Series(kumo_thickness).rolling(window=20, min_periods=20).mean().values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.values.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish.values.astype(float))
    kumo_thickness_ma_aligned = align_htf_to_ltf(prices, df_1d, kumo_thickness_ma)
    
    # Get 12h data for volume confirmation and additional trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    close_12h = df_12h['close'].values
    
    # Volume confirmation: current 12h volume > 1.5x 20-period mean
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_confirm_12h = volume_12h > (vol_ma_20_12h * 1.5)
    
    # 12h trend filter: price above/below 20-period EMA
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_filter_12h = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume spike on 6h timeframe
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume > (vol_ma_20_6h * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations
    start_idx = 120
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(kumo_twist_bullish_aligned[i]) or
            np.isnan(kumo_twist_bearish_aligned[i]) or
            np.isnan(kumo_thickness_ma_aligned[i]) or
            np.isnan(trend_filter_12h[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_kumo = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross signals
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Long: Kumo twist bullish OR (price above Kumo AND TK cross bullish) with volume confirmation and 12h trend alignment
            long_condition = (
                (kumo_twist_bullish_aligned[i] > 0.5) or  # Fresh Kumo twist bullish
                (
                    close[i] > upper_kumo and  # Price above cloud
                    tk_cross_bullish and       # TK cross bullish
                    vol_spike_6h[i] and        # Volume spike on 6h
                    close[i] > trend_filter_12h[i] and  # 12h uptrend
                    kumo_thickness_ma_aligned[i] > 0  # Cloud has thickness (not too thin)
                )
            )
            
            # Short: Kumo twist bearish OR (price below Kumo AND TK cross bearish) with volume confirmation and 12h trend alignment
            short_condition = (
                (kumo_twist_bearish_aligned[i] > 0.5) or  # Fresh Kumo twist bearish
                (
                    close[i] < lower_kumo and   # Price below cloud
                    tk_cross_bearish and        # TK cross bearish
                    vol_spike_6h[i] and         # Volume spike on 6h
                    close[i] < trend_filter_12h[i] and  # 12h downtrend
                    kumo_thickness_ma_aligned[i] > 0   # Cloud has thickness
                )
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price falls below Kumo OR TK cross bearish with volume confirmation
            exit_condition = (
                close[i] < lower_kumo or           # Price breaks below cloud
                (tk_cross_bearish and vol_spike_6h[i])  # TK cross bearish with volume
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price rises above Kumo OR TK cross bullish with volume confirmation
            exit_condition = (
                close[i] > upper_kumo or            # Price breaks above cloud
                (tk_cross_bullish and vol_spike_6h[i])  # TK cross bullish with volume
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_TK_Cross_v1"
timeframe = "6h"
leverage = 1.0