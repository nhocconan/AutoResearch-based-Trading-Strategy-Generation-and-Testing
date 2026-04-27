#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud breakout with weekly trend filter and volume confirmation.
Trades breakouts above/below the Kumo (cloud) when the weekly trend agrees and volume exceeds 1.5x the 60-period average.
The Ichimoku Cloud provides dynamic support/resistance, while the weekly trend filter ensures we trade in the direction of higher timeframe momentum.
Volume confirmation reduces false breakouts. Designed for 6H timeframe to capture medium-term moves with lower trade frequency.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (standard settings: 9, 26, 52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(df_1d['close'].values).shift(26)
    
    # Align Ichimoku components to 6-hour timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span.values)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 60-period volume average for confirmation
    vol_ma_60 = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Ichimoku components, weekly EMA, and volume MA
    start_idx = max(52 + 26, 20, 60)  # Senkou B shift + weekly EMA + volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma_60[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_60[i]
        
        # Ichimoku Cloud boundaries (top and bottom of Kumo)
        top_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        bottom_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Weekly trend: price above/below weekly EMA
        weekly_trend_up = price_now > ema_20_1w_aligned[i]
        weekly_trend_down = price_now < ema_20_1w_aligned[i]
        
        # Volume confirmation: volume > 1.5x 60-period average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Chikou confirmation: lagging span above/below price from 26 periods ago
        chikou_confirm_long = chikou_span_aligned[i] > close[i - 26] if i >= 26 else False
        chikou_confirm_short = chikou_span_aligned[i] < close[i - 26] if i >= 26 else False
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above cloud, weekly uptrend, volume confirmation, Chikou confirms
            if (price_now > top_cloud and 
                weekly_trend_up and 
                vol_filter and 
                chikou_confirm_long):
                signals[i] = size
                position = 1
            # Short: price breaks below cloud, weekly downtrend, volume confirmation, Chikou confirms
            elif (price_now < bottom_cloud and 
                  weekly_trend_down and 
                  vol_filter and 
                  chikou_confirm_short):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below cloud or weekly trend turns down
            if price_now < bottom_cloud or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above cloud or weekly trend turns up
            if price_now > top_cloud or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0