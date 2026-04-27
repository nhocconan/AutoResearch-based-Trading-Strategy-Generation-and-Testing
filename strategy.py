#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation.
Long when Tenkan-sen crosses above Kijun-sen AND price above Kumo (cloud) AND price > 1-day EMA50.
Short when Tenkan-sen crosses below Kijun-sen AND price below Kumo AND price < 1-day EMA50.
Ichimoku provides multi-layer trend confirmation; daily EMA filters higher-timeframe trend;
volume confirms institutional participation. Target: 12-35 trades/year per symbol.
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
    
    # Calculate daily EMA50 for trend
    daily_close = df_1d['close'].values
    ema_50_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_50_1d.fill(np.nan)
    if len(daily_close) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(daily_close[:50])
        for i in range(50, len(daily_close)):
            ema_50_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full_like(high, np.nan)
    period9_low = np.full_like(low, np.nan)
    for i in range(8, len(high)):
        period9_high[i] = np.max(high[i-8:i+1])
        period9_low[i] = np.min(low[i-8:i+1])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full_like(high, np.nan)
    period26_low = np.full_like(low, np.nan)
    for i in range(25, len(high)):
        period26_high[i] = np.max(high[i-25:i+1])
        period26_low[i] = np.min(low[i-25:i+1])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.full_like(high, np.nan)
    period52_low = np.full_like(low, np.nan)
    for i in range(51, len(high)):
        period52_high[i] = np.max(high[i-51:i+1])
        period52_low[i] = np.min(low[i-51:i+1])
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get daily data for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = np.empty_like(vol_1d, dtype=np.float64)
    vol_ma_20_1d.fill(np.nan)
    for i in range(19, len(vol_1d)):
        vol_ma_20_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Ichimoku (52), daily EMA50 (50), daily volume MA20 (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Kumo (Cloud) boundaries
        upper_cloud = max(senkou_a, senkou_b)
        lower_cloud = min(senkou_a, senkou_b)
        
        # Price above/below cloud
        above_cloud = price_now > upper_cloud
        below_cloud = price_now < lower_cloud
        
        # Tenkan/Kijun crossover
        if i > start_idx:
            tenkan_prev = tenkan_aligned[i-1]
            kijun_prev = kijun_aligned[i-1]
            tk_cross_above = (tenkan_prev <= kijun_prev) and (tenkan > kijun)
            tk_cross_below = (tenkan_prev >= kijun_prev) and (tenkan < kijun)
        else:
            tk_cross_above = False
            tk_cross_below = False
        
        # Volume filter: volume > 1.2x daily average
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Daily close price for trend comparison
        daily_close_price = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_price)
        if np.isnan(daily_close_aligned[i]):
            signals[i] = 0.0
            continue
        daily_close_val = daily_close_aligned[i]
        
        if position == 0:
            # Long conditions: TK cross bullish + price above cloud + price > daily EMA50 + volume
            if tk_cross_above and above_cloud and (daily_close_val > ema_trend) and vol_filter:
                signals[i] = size
                position = 1
            # Short conditions: TK cross bearish + price below cloud + price < daily EMA50 + volume
            elif tk_cross_below and below_cloud and (daily_close_val < ema_trend) and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross bearish OR price below cloud OR trend turns bearish
            if tk_cross_below or (price_now < upper_cloud) or (daily_close_val < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross bullish OR price above cloud OR trend turns bullish
            if tk_cross_above or (price_now > lower_cloud) or (daily_close_val > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0