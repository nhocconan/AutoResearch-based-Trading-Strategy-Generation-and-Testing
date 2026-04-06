#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Volume Spike + Trend Filter
Hypothesis: Ichimoku signals (TK cross + price above/below cloud) filtered by 1d trend and volume spikes capture high-probability trend continuation moves. Works in bull/bear by requiring alignment with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line)
    tenkan_sen = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        period_high = np.max(high[i-tenkan_period+1:i+1])
        period_low = np.min(low[i-tenkan_period+1:i+1])
        tenkan_sen[i] = (period_high + period_low) / 2
    
    # Calculate Kijun-sen (Base Line)
    kijun_sen = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        period_high = np.max(high[i-kijun_period+1:i+1])
        period_low = np.min(low[i-kijun_period+1:i+1])
        kijun_sen[i] = (period_high + period_low) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B)
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        period_high = np.max(high[i-senkou_span_b_period+1:i+1])
        period_low = np.min(low[i-senkou_span_b_period+1:i+1])
        senkou_span_b[i] = (period_high + period_low) / 2
    
    # 1d trend filter using EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = close_1d[49]
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 49) / 50
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter (24-period average ~ 4 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 52  # For Senkou Span B
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Ichimoku signals
        price_above_cloud = close[i] > max(senkou_span_a[i], senkou_span_b[i])
        price_below_cloud = close[i] < min(senkou_span_a[i], senkou_span_b[i])
        tk_bullish = tenkan_sen[i] > kijun_sen[i]
        tk_bearish = tenkan_sen[i] < kijun_sen[i]
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # 1d trend filter
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: TK cross bearish OR price below cloud
            if tk_bearish or price_below_cloud:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: TK cross bullish OR price above cloud
            if tk_bullish or price_above_cloud:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Ichimoku signal + volume + 1d trend filter
            # Minimum holding period: only allow new entry after 24 bars flat
            if bars_since_entry >= 24:
                bull_setup = price_above_cloud and tk_bullish and volume_filter and trend_up
                bear_setup = price_below_cloud and tk_bearish and volume_filter and trend_down
                
                if bull_setup:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_setup:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals