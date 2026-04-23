#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
Long when price breaks above Kumo (cloud) AND Tenkan > Kijun AND 1d EMA50 rising AND volume > 2x 20-period MA.
Short when price breaks below Kumo AND Tenkan < Kijun AND 1d EMA50 falling AND volume > 2x 20-period MA.
Exit when price re-enters Kumo or Tenkan/Kijun cross reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Ichimoku provides dynamic support/resistance via cloud, works in both bull/bear markets by following higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = np.full(n, np.nan)
    for i in range(kijun_period - 1, n - kijun_period + 1):
        senkou_span_a[i + kijun_period] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n - kijun_period + 1):
        senkou_span_b[i + kijun_period] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, 50, 20) + kijun_period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a = senkou_span_a[i]
        senkou_b = senkou_span_b[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Kumo (cloud) boundaries
        upper_kumo = max(senkou_a, senkou_b)
        lower_kumo = min(senkou_a, senkou_b)
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 2x 20-period MA (higher threshold to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Price breaks above Kumo AND Tenkan > Kijun AND EMA50 rising AND volume filter
            if price > upper_kumo and tenkan_val > kijun_val and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Kumo AND Tenkan < Kijun AND EMA50 falling AND volume filter
            elif price < lower_kumo and tenkan_val < kijun_val and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price re-enters Kumo OR Tenkan < Kijun (cross reversal)
                if price < upper_kumo or tenkan_val < kijun_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: price re-enters Kumo OR Tenkan > Kijun (cross reversal)
                if price > lower_kumo or tenkan_val > kijun_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Kumo_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0