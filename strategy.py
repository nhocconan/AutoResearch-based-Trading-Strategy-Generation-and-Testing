#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
Long when price breaks above Ichimoku cloud (Senkou Span A/B) AND 1d EMA50 rising AND volume > 1.5x 20-period MA.
Short when price breaks below Ichimoku cloud AND 1d EMA50 falling AND volume > 1.5x 20-period MA.
Exit when price re-enters the cloud or 1d EMA50 reverses.
Ichimoku provides dynamic support/resistance, 1d EMA50 filters major trend, volume confirms momentum.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Works in bull (trend-aligned breakouts) and bear (volume spikes on breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = np.full(n, np.nan)
    for i in range(period_tenkan - 1, n):
        tenkan_sen[i] = (np.max(high[i-period_tenkan+1:i+1]) + np.min(low[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = np.full(n, np.nan)
    for i in range(period_kijun - 1, n):
        kijun_sen[i] = (np.max(high[i-period_kijun+1:i+1]) + np.min(low[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    senkou_span_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + period_kijun  # shift 26 periods ahead
            if idx < n:
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = np.full(n, np.nan)
    for i in range(period_senkou_b - 1, n):
        span_b_val = (np.max(high[i-period_senkou_b+1:i+1]) + np.min(low[i-period_senkou_b+1:i+1])) / 2
        idx = i + period_kijun  # shift 26 periods ahead
        if idx < n:
            senkou_span_b[idx] = span_b_val
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need: Senkou Span B (52-period) + 26 shift + EMA50 + volume MA
    start_idx = max(52 + 26 - 1, 50, 20)  # Senkou B calculation + shift, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_cloud = max(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = min(senkou_span_a[i], senkou_span_b[i])
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above cloud AND EMA50 rising AND volume filter
            if price > upper_cloud and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below cloud AND EMA50 falling AND volume filter
            elif price < lower_cloud and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price re-enters cloud (below upper cloud) OR EMA50 starts falling
                if price < upper_cloud or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price re-enters cloud (above lower cloud) OR EMA50 starts rising
                if price > lower_cloud or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_Breakout_1dEMA50_Trend_VolumeFilter"
timeframe = "6h"
leverage = 1.0