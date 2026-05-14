#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout + Volume Confirmation + 12h EMA50 Trend Filter
Hypothesis: Ichimoku cloud provides dynamic support/resistance with forward-looking cloud.
Breakouts above/below cloud with volume spike and 12h EMA50 trend alignment capture strong moves.
Works in bull/bear via 12h EMA50 trend filter (only trade in trend direction).
Designed for 50-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max()
    lowest_tenkan = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun, min_periods=kijun).max()
    lowest_kijun = pd.Series(low).rolling(window=kijun, min_periods=kijun).min()
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    highest_senkou = pd.Series(high).rolling(window=senkou, min_periods=senkou).max()
    lowest_senkou = pd.Series(low).rolling(window=senkou, min_periods=senkou).min()
    senkou_b = ((highest_senkou + lowest_senkou) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA50 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50_12h = calculate_ema(df_12h['close'].values, 50)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ichimoku Cloud on 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Cloud top and bottom (Senkou A and B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) + EMA (50) + volume MA (30)
    start_idx = max(52, 50, 30) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Ichimoku signals
        price_above_cloud = curr_close > cloud_top[i]
        price_below_cloud = curr_close < cloud_bottom[i]
        bullish_tk_cross = tenkan[i] > kijun[i]
        bearish_tk_cross = tenkan[i] < kijun[i]
        
        if position == 0:
            # Look for entry signals - require: Ichimoku breakout + volume spike + 12h EMA50 trend alignment
            long_entry = price_above_cloud and bullish_tk_cross and vol_ma[i] > 0 and volume_spike[i] and (curr_close > ema_50_12h_aligned[i])
            short_entry = price_below_cloud and bearish_tk_cross and vol_ma[i] > 0 and volume_spike[i] and (curr_close < ema_50_12h_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below cloud or TK cross turns bearish
            if curr_close < cloud_top[i] or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above cloud or TK cross turns bullish
            if curr_close > cloud_bottom[i] or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_Volume_12hEMA50_Trend"
timeframe = "6h"
leverage = 1.0