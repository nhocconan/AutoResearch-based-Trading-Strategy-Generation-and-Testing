#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Filter
Hypothesis: 6h Ichimoku TK cross with Kumo twist (Senkou A/B cross) as momentum signal,
filtered by 1d trend (price >/<- EMA50). Long when TK crosses above AND price > Kumo AND 1d uptrend.
Short when TK crosses below AND price < Kumo AND 1d downtrend. Uses discrete sizing (0.25) to minimize fees.
Ichimoku works in trending markets; 1d filter avoids counter-trend trades in chop.
Target: 12-30 trades/year on 6h. Works in bull via trend continuation, in bear via avoiding false signals.
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
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations + displacement
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + displacement
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Kumo twist: Senkou A crossing Senkou B
        # Kumo twist bullish: Senkou A > Senkou B (and was <=)
        # Kumo twist bearish: Senkou A < Senkou B (and was >=)
        if i == 0:
            senkou_a_prev = senkou_a[i]
            senkou_b_prev = senkou_b[i]
        else:
            senkou_a_prev = senkou_a[i-1]
            senkou_b_prev = senkou_b[i-1]
        
        kumo_twist_bullish = (senkou_a[i] > senkou_b[i]) and (senkou_a_prev <= senkou_b_prev)
        kumo_twist_bearish = (senkou_a[i] < senkou_b[i]) and (senkou_a_prev >= senkou_b_prev)
        
        # TK cross: Tenkan crossing Kijun
        # TK cross bullish: Tenkan > Kijun (and was <=)
        # TK cross bearish: Tenkan < Kijun (and was >=)
        if i == 0:
            tenkan_prev = tenkan[i]
            kijun_prev = kijun[i]
        else:
            tenkan_prev = tenkan[i-1]
            kijun_prev = kijun[i-1]
        
        tk_cross_bullish = (tenkan[i] > kijun[i]) and (tenkan_prev <= kijun_prev)
        tk_cross_bearish = (tenkan[i] < kijun[i]) and (tenkan_prev >= kijun_prev)
        
        # Price relative to Kumo (cloud)
        # Price above Kumo: price > max(senkou_a, senkou_b)
        # Price below Kumo: price < min(senkou_a, senkou_b)
        kumo_top = np.maximum(senkou_a[i], senkou_b[i])
        kumo_bottom = np.minimum(senkou_a[i], senkou_b[i])
        price_above_kumo = close[i] > kumo_top
        price_below_kumo = close[i] < kumo_bottom
        
        if position == 0:
            # Long: TK cross bullish AND Kumo twist bullish AND price above Kumo AND 1d uptrend
            long_signal = tk_cross_bullish and kumo_twist_bullish and price_above_kumo and (close[i] > ema_50_1d_aligned[i])
            # Short: TK cross bearish AND Kumo twist bearish AND price below Kumo AND 1d downtrend
            short_signal = tk_cross_bearish and kumo_twist_bearish and price_below_kumo and (close[i] < ema_50_1d_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when TK cross bearish OR price falls below Kumo bottom OR 1d trend turns down
            exit_signal = tk_cross_bearish or (close[i] < kumo_bottom) or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when TK cross bullish OR price rises above Kumo top OR 1d trend turns up
            exit_signal = tk_cross_bullish or (close[i] > kumo_top) or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0