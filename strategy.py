#!/usr/bin/env python3
"""
6h_1dIchimoku_TK_Cross_WeeklyTrend
6h strategy using Ichimoku Tenkan/Kijun cross from daily timeframe with weekly trend filter.
- Long: TK cross bullish (Tenkan > Kijun) on 1D + price > Kumo (cloud) on 1D + weekly EMA200 > weekly EMA200[5] (uptrend)
- Short: TK cross bearish (Tenkan < Kijun) on 1D + price < Kumo (cloud) on 1D + weekly EMA200 < weekly EMA200[5] (downtrend)
- Exit: Opposite TK cross or trend reversal
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Ichimoku provides institutional-grade support/resistance; weekly EMA trend filter avoids counter-trend whipsaws
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
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # For cloud top/bottom in Ichimoku, we need to plot Senkou Span A/B 26 periods ahead
    # But for current cloud, we use current Senkou Span A/B values
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # TK Cross signals
    tk_cross_bullish = tenkan_sen > kijun_sen
    tk_cross_bearish = tenkan_sen < kijun_sen
    
    # Price relative to cloud
    price_above_kumo = close_1d > kumo_top
    price_below_kumo = close_1d < kumo_bottom
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    tk_cross_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_bullish)
    tk_cross_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_bearish)
    price_above_kumo_aligned = align_htf_to_ltf(prices, df_1d, price_above_kumo)
    price_below_kumo_aligned = align_htf_to_ltf(prices, df_1d, price_below_kumo)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Weekly EMA200 slope (5-period change) for trend strength
    ema_200_1w_shift = np.roll(ema_200_1w_aligned, 5)
    ema_200_1w_shift[:5] = np.nan
    ema_200_slope = ema_200_1w_aligned - ema_200_1w_shift
    ema_200_uptrend = ema_200_slope > 0
    ema_200_downtrend = ema_200_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 200)  # need enough for Ichomoku and EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(tk_cross_bullish_aligned[i]) or np.isnan(tk_cross_bearish_aligned[i]) or
            np.isnan(price_above_kumo_aligned[i]) or np.isnan(price_below_kumo_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(ema_200_slope[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from weekly EMA200 slope
        uptrend = ema_200_uptrend[i]
        downtrend = ema_200_downtrend[i]
        
        if position == 0:
            # Long: bullish TK cross + price above cloud + weekly uptrend
            if tk_cross_bullish_aligned[i] and price_above_kumo_aligned[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below cloud + weekly downtrend
            elif tk_cross_bearish_aligned[i] and price_below_kumo_aligned[i] and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish TK cross or price below cloud or weekly downtrend
            if tk_cross_bearish_aligned[i] or not price_above_kumo_aligned[i] or downtrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish TK cross or price above cloud or weekly uptrend
            if tk_cross_bullish_aligned[i] or price_below_kumo_aligned[i] or uptrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dIchimoku_TK_Cross_WeeklyTrend"
timeframe = "6h"
leverage = 1.0