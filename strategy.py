#!/usr/bin/env python3
"""
6h_Ichimoku_CloudBreakout_TrendFilter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. Buy when price breaks above cloud with bullish TK cross and bullish weekly trend (weekly close > weekly EMA50). Sell when price breaks below cloud with bearish TK cross and bearish weekly trend. Weekly trend filter prevents counter-trend trades in ranging/bear markets. Designed for 6H timeframe to capture multi-day moves with tight entries (~15-25 trades/year).
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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past 9 periods
    highest_high_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_low_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over past 26 periods
    highest_high_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_low_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past 52 periods
    highest_high_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_b = (highest_high_52 + lowest_low_52) / 2
    
    # Cloud top/bottom (using current Senkou spans)
    # Note: Senkou spans are shifted forward by kijun_period (26) in Ichimoku,
    # but for cloud breakout we use current Senkou A/B as cloud boundaries
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK Cross signals
    tk_cross_bullish = tenkan > kijun
    tk_cross_bearish = tenkan < kijun
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Weekly trend filter: weekly close > weekly EMA50 for uptrend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_series_1w = pd.Series(close_1w)
    ema50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    weekly_uptrend = close_1w > ema50_1w_aligned  # using aligned weekly data
    
    # For weekly downtrend, we need the opposite
    weekly_downtrend = close_1w < ema50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    # Start after all indicators are valid
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(weekly_uptrend[i]) or np.isnan(weekly_downtrend[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        bullish_breakout = price_above_cloud[i] and tk_cross_bullish[i] and weekly_uptrend[i]
        bearish_breakout = price_below_cloud[i] and tk_cross_bearish[i] and weekly_downtrend[i]
        
        if position == 0:
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below cloud OR TK cross turns bearish
            if price_below_cloud[i] or tk_cross_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above cloud OR TK cross turns bullish
            if price_above_cloud[i] or tk_cross_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_CloudBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0