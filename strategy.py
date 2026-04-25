#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dEMA50_Trend_VolumeSpike
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. Breakouts above/below cloud with volume spike and 1d EMA50 trend alignment capture strong moves. Works in bull/bear via 1d EMA50 trend filter (only trade in trend direction). Designed for 50-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter and Ichimoku (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku from 1d data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52) + EMA (50) + volume MA (20)
    start_idx = max(52, 50, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Ichimoku cloud: Senkou A and Senkou B form the cloud
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Look for entry signals - require: Cloud breakout + volume spike + 1d EMA50 trend alignment
            # Long: price breaks above cloud top in uptrend
            long_entry = (curr_close > cloud_top) and vol_ma[i] > 0 and volume_spike[i] and (curr_close > ema_50_1d_aligned[i])
            # Short: price breaks below cloud bottom in downtrend
            short_entry = (curr_close < cloud_bottom) and vol_ma[i] > 0 and volume_spike[i] and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below cloud bottom or trend turns bearish
            if curr_close < cloud_bottom or curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above cloud top or trend turns bullish
            if curr_close > cloud_top or curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0