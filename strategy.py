#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter_v2
Hypothesis: Use Ichimoku cloud from 1d for trend filter, with TK cross on 6h for entry timing and volume confirmation. Only take longs when price is above 1d cloud and TK cross bullish, shorts when below cloud and TK cross bearish. Add 1w trend filter to avoid counter-trend trades in strong weekly trends. Discrete sizing 0.25. Target 12-25 trades/year to minimize fee drag while capturing strong trending moves with multi-timeframe confluence.
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
    volume = prices['volume'].values
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    if len(high_1d) >= period_tenkan:
        tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() +
                     pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    else:
        tenkan_sen = np.full_like(close_1d, np.nan)
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    if len(high_1d) >= period_kijun:
        kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() +
                    pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    else:
        kijun_sen = np.full_like(close_1d, np.nan)
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    if len(high_1d) >= period_senkou_b:
        senkou_span_b = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() +
                        pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    else:
        senkou_span_b = np.full_like(close_1d, np.nan)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = np.concatenate([np.full(26, np.nan), close_1d[:-26]]) if len(close_1d) > 26 else np.full_like(close_1d, np.nan)
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values, additional_delay_bars=26)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span, additional_delay_bars=26)
    
    # 1w EMA50 for weekly trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # TK Cross on 6h: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_bullish = tenkan_sen_aligned > kijun_sen_aligned
    tk_cross_bearish = tenkan_sen_aligned < kijun_sen_aligned
    
    # Price relative to cloud: above/both Senkou spans
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Weekly trend filter: price above/below 1w EMA50
    weekly_uptrend = close > ema_50_1w_aligned
    weekly_downtrend = close < ema_50_1w_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52) + volume MA (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price above cloud + TK cross bullish + weekly uptrend + volume confirmation
            long_signal = (price_above_cloud[i] and 
                          tk_cross_bullish[i] and 
                          weekly_uptrend[i] and 
                          volume_confirmed[i])
            # Short: price below cloud + TK cross bearish + weekly downtrend + volume confirmation
            short_signal = (price_below_cloud[i] and 
                           tk_cross_bearish[i] and 
                           weekly_downtrend[i] and 
                           volume_confirmed[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below cloud OR TK cross turns bearish
            if not (price_above_cloud[i] and tk_cross_bullish[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above cloud OR TK cross turns bullish
            if not (price_below_cloud[i] and tk_cross_bearish[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter_v2"
timeframe = "6h"
leverage = 1.0