#!/usr/bin/env python3
"""
6h_1d_1w_Ichimoku_TK_Cross_Cloud_Filter_v1
Hypothesis: Ichimoku TK cross with cloud filter from daily timeframe and weekly trend filter.
Long when TK crosses above Kijun and price above daily cloud + weekly trend up.
Short when TK crosses below Kijun and price below daily cloud + weekly trend down.
Exit when TK crosses back or price exits cloud.
Targets 15-30 trades/year per symbol by requiring multiple confluence factors.
Works in bull/bear by following weekly trend and using cloud as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past 9 periods
    tenkan_sen = []
    for i in range(len(high_1d)):
        if i < tenkan_period - 1:
            tenkan_sen.append(np.nan)
        else:
            window_high = high_1d[i-tenkan_period+1:i+1]
            window_low = low_1d[i-tenkan_period+1:i+1]
            tenkan_sen.append((np.nanmax(window_high) + np.nanmin(window_low)) / 2)
    tenkan_sen = np.array(tenkan_sen)
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over past 26 periods
    kijun_sen = []
    for i in range(len(high_1d)):
        if i < kijun_period - 1:
            kijun_sen.append(np.nan)
        else:
            window_high = high_1d[i-kijun_period+1:i+1]
            window_low = low_1d[i-kijun_period+1:i+1]
            kijun_sen.append((np.nanmax(window_high) + np.nanmin(window_low)) / 2)
    kijun_sen = np.array(kijun_sen)
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past 52 periods shifted forward 26
    senkou_span_b = []
    for i in range(len(high_1d)):
        if i < senkou_span_b_period - 1:
            senkou_span_b.append(np.nan)
        else:
            window_high = high_1d[i-senkou_span_b_period+1:i+1]
            window_low = low_1d[i-senkou_span_b_period+1:i+1]
            senkou_span_b.append((np.nanmax(window_high) + np.nanmin(window_low)) / 2)
    senkou_span_b = np.array(senkou_span_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA34 on weekly for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK crossover signals
        tk_cross_above = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_below = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # Weekly trend filter
        if i >= 101:
            weekly_uptrend = ema34_1w_aligned[i] > ema34_1w_aligned[i-1]
            weekly_downtrend = ema34_1w_aligned[i] < ema34_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: TK cross above + price above cloud + weekly uptrend
            if tk_cross_above and price_above_cloud and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below + price below cloud + weekly downtrend
            elif tk_cross_below and price_below_cloud and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross below OR price drops below cloud
            if tk_cross_below or price < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross above OR price rises above cloud
            if tk_cross_above or price > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_Ichimoku_TK_Cross_Cloud_Filter_v1"
timeframe = "6h"
leverage = 1.0