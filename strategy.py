#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1-day trend filter and volume confirmation.
Long when price > Kumo cloud + Tenkan > Kijun + price > 1-day EMA34 + volume spike.
Short when price < Kumo cloud + Tenkan < Kijun + price < 1-day EMA34 + volume spike.
Ichimoku provides dynamic support/resistance and momentum, daily trend filters direction,
volume confirms institutional participation. Target: 20-40 trades/year per symbol.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    daily_close = df_1d['close'].values
    ema_34_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.empty_like(high, dtype=np.float64)
    tenkan_sen.fill(np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan_sen[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.empty_like(high, dtype=np.float64)
    kijun_sen.fill(np.nan)
    for i in range(kijun_period - 1, n):
        kijun_sen[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = np.empty_like(high, dtype=np.float64)
    senkou_span_a.fill(np.nan)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = np.empty_like(high, dtype=np.float64)
    senkou_span_b.fill(np.nan)
    for i in range(senkou_span_b_period - 1, n):
        senkou_span_b[i] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods back
    chikou_span = np.empty_like(close, dtype=np.float64)
    chikou_span.fill(np.nan)
    for i in range(kijun_period - 1, n):
        chikou_span[i] = close[i - kijun_period + 1]
    
    # Calculate cloud boundaries (future Senkou spans shifted forward)
    # For simplicity, we use current Senkou spans as cloud (standard Ichimoku cloud)
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.empty_like(high, dtype=np.float64)
    cloud_bottom = np.empty_like(high, dtype=np.float64)
    for i in range(n):
        if not np.isnan(senkou_span_a[i]) and not np.isnan(senkou_span_b[i]):
            cloud_top[i] = max(senkou_span_a[i], senkou_span_b[i])
            cloud_bottom[i] = min(senkou_span_a[i], senkou_span_b[i])
        else:
            cloud_top[i] = np.nan
            cloud_bottom[i] = np.nan
    
    # Volume filter: volume > 1.5x average (calculated from 6h volume MA20)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Ichimoku components (52 periods for Senkou B)
    start_idx = senkou_span_b_period - 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Daily close price for trend comparison
        daily_close_price = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_price)
        if np.isnan(daily_close_aligned[i]):
            signals[i] = 0.0
            continue
        daily_close_val = daily_close_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        # Ichimoku signals
        price_above_cloud = price_now > cloud_top_val
        price_below_cloud = price_now < cloud_bottom_val
        tenkan_above_kijun = tenkan > kijun
        tenkan_below_kijun = tenkan < kijun
        
        if position == 0:
            # Bull conditions: price > cloud + Tenkan > Kijun + bull trend + volume
            if (price_above_cloud and tenkan_above_kijun and 
                daily_close_val > ema_trend and vol_filter):
                signals[i] = size
                position = 1
            # Bear conditions: price < cloud + Tenkan < Kijun + bear trend + volume
            elif (price_below_cloud and tenkan_below_kijun and 
                  daily_close_val < ema_trend and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < cloud or Tenkan < Kijun or trend turns bear
            if (price_now < cloud_top_val or tenkan < kijun or 
                daily_close_val < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > cloud or Tenkan > Kijun or trend turns bull
            if (price_now > cloud_bottom_val or tenkan > kijun or 
                daily_close_val > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0