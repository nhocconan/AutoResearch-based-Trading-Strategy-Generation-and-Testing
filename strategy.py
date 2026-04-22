#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud with 1-week trend filter.
Long when price > Kumo (cloud) and Tenkan-sen > Kijun-sen, with 1-week trend bullish.
Short when price < Kumo and Tenkan-sen < Kijun-sen, with 1-week trend bearish.
Ichimoku provides dynamic support/resistance and momentum signals. Weekly trend filter
ensures alignment with higher timeframe direction, reducing whipsaws in sideways markets.
Works in bull markets by catching uptrends and in bear markets by avoiding false longs.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        period_high = np.max(high[i - tenkan_period + 1:i + 1])
        period_low = np.min(low[i - tenkan_period + 1:i + 1])
        tenkan_sen[i] = (period_high + period_low) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        period_high = np.max(high[i - kijun_period + 1:i + 1])
        period_low = np.min(low[i - kijun_period + 1:i + 1])
        kijun_sen[i] = (period_high + period_low) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = np.full(n, np.nan)
    valid_mask = ~(np.isnan(tenkan_sen) | np.isnan(kijun_sen))
    senkou_span_a[valid_mask] = (tenkan_sen[valid_mask] + kijun_sen[valid_mask]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        period_high = np.max(high[i - senkou_span_b_period + 1:i + 1])
        period_low = np.min(low[i - senkou_span_b_period + 1:i + 1])
        senkou_span_b[i] = (period_high + period_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward by 26 periods
    senkou_span_a_shifted = np.full(n, np.nan)
    senkou_span_b_shifted = np.full(n, np.nan)
    for i in range(n - kijun_period):
        senkou_span_a_shifted[i + kijun_period] = senkou_span_a[i]
        senkou_span_b_shifted[i + kijun_period] = senkou_span_b[i]
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly trend: price above/below 20-period EMA
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if Ichimoku data not ready
        if np.isnan(senkou_span_a_shifted[i]) or np.isnan(senkou_span_b_shifted[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_shifted[i], senkou_span_b_shifted[i])
        cloud_bottom = min(senkou_span_a_shifted[i], senkou_span_b_shifted[i])
        
        if position == 0:
            # Long: Price above cloud, bullish TK cross, and weekly uptrend
            if (close[i] > cloud_top and 
                tenkan_sen[i] > kijun_sen[i] and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, bearish TK cross, and weekly downtrend
            elif (close[i] < cloud_bottom and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below cloud or bearish TK cross
                if close[i] < cloud_bottom or tenkan_sen[i] < kijun_sen[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above cloud or bullish TK cross
                if close[i] > cloud_top or tenkan_sen[i] > kijun_sen[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0