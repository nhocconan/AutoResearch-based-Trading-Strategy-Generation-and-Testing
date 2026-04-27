#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with 1d Trend Filter and Volume Confirmation.
Long when price breaks above Kumo cloud + Tenkan/Kijun bullish cross + 1d trend up + volume spike.
Short when price breaks below Kumo cloud + Tenkan/Kijun bearish cross + 1d trend down + volume spike.
Exit when price re-enters Kumo cloud or trend changes.
Designed for low frequency (12-37 trades/year) to minimize fee drift.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend
    daily_close = df_1d['close'].values
    ema_50_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_50_1d.fill(np.nan)
    if len(daily_close) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(daily_close[:50])
        for i in range(50, len(daily_close)):
            ema_50_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_period = 9
    tenkan_sen = np.empty_like(high, dtype=np.float64)
    tenkan_sen.fill(np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan_sen[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_period = 26
    kijun_sen = np.empty_like(high, dtype=np.float64)
    kijun_sen.fill(np.nan)
    for i in range(kijun_period - 1, n):
        kijun_sen[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    senkou_span_a = np.empty_like(high, dtype=np.float64)
    senkou_span_a.fill(np.nan)
    for i in range(n):
        if i + kijun_period < n and not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i + kijun_period] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    senkou_b_period = 52
    senkou_span_b = np.empty_like(high, dtype=np.float64)
    senkou_span_b.fill(np.nan)
    for i in range(senkou_b_period - 1, n):
        senkou_span_b[i + kijun_period] = (np.max(high[i-senkou_b_period+1:i+1]) + np.min(low[i-senkou_b_period+1:i+1])) / 2
    
    # Current Kumo cloud boundaries (use Senkou spans shifted back)
    senkou_a_current = np.empty_like(high, dtype=np.float64)
    senkou_b_current = np.empty_like(high, dtype=np.float64)
    senkou_a_current.fill(np.nan)
    senkou_b_current.fill(np.nan)
    for i in range(kijun_period, n):
        if i - kijun_period >= 0:
            senkou_a_current[i] = senkou_span_a[i]
            senkou_b_current[i] = senkou_span_b[i]
    
    # Kumo cloud top and bottom
    kumo_top = np.maximum(senkou_a_current, senkou_b_current)
    kumo_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # Volume filter: volume > 1.5x average (60-period)
    vol_ma_60 = np.empty_like(volume, dtype=np.float64)
    vol_ma_60.fill(np.nan)
    for i in range(59, n):
        vol_ma_60[i] = np.mean(volume[i-59:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Ichimoku (52+26=78 periods) and daily EMA (50 periods)
    start_idx = max(52 + kijun_period - 1, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_60[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        daily_trend = ema_50_1d_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_60[i]
        
        # Ichimoku signals
        price_above_kumo = price_now > kumo_top_val
        price_below_kumo = price_now < kumo_bottom_val
        tk_bullish_cross = tenkan > kijun
        tk_bearish_cross = tenkan < kijun
        
        if position == 0:
            # Bull: price above Kumo + TK bullish cross + daily trend up + volume
            if price_above_kumo and tk_bullish_cross and price_now > daily_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price below Kumo + TK bearish cross + daily trend down + volume
            elif price_below_kumo and tk_bearish_cross and price_now < daily_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Kumo or daily trend turns down
            if price_now < kumo_top_val or price_now < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Kumo or daily trend turns up
            if price_now > kumo_bottom_val or price_now > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_KumoBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0