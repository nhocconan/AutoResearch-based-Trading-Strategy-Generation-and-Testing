#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend_v1
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun cross with price above/below cloud from 1d, 
filtered by weekly trend (price vs weekly EMA50). Targets 12-30 trades/year by requiring 
confluence of momentum (TK cross), structure (cloud position), and higher timeframe trend.
Works in bull/bear via trend filter and cloud acting as dynamic support/resistance.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for Ichimoku components (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_high_9 = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_low_9 = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_high_26 = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_low_26 = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted forward 26
    highest_high_52 = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_52 = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (52 + 26 offset for Senkou)
    start_idx = 78  # 52 + 26
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Price above/below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Tenkan-Kijun cross
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals: TK cross in direction of trend, price outside cloud
            # Long: bullish TK cross, price above cloud, uptrend
            long_signal = tk_cross_up and price_above_cloud and uptrend
            # Short: bearish TK cross, price below cloud, downtrend
            short_signal = tk_cross_down and price_below_cloud and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on bearish TK cross or price re-enters cloud
            if tk_cross_down or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on bullish TK cross or price re-enters cloud
            if tk_cross_up or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend_v1"
timeframe = "6h"
leverage = 1.0