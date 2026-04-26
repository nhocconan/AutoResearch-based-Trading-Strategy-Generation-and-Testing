#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_VolumeConfirm
Hypothesis: Trade 6h Ichimoku cloud breakouts in direction of 1d EMA50 trend with volume confirmation.
Uses 6h primary timeframe for lower trade frequency. Ichimoku components calculated on 6h: Tenkan (9), Kijun (26), Senkou Span A/B (52 displacement), Chikou (26 lag).
Trend filter: 1d EMA50 for higher timeframe alignment. Volume spike on 6h confirms breakout.
Works in bull/bear via trend filter + volume confirmation. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = (pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe (account for forward shift)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Volume confirmation: volume > 2.0x 20-period average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), Ichimoku (52), volume MA (20)
    start_idx = max(50, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: price breaks above cloud + price above 1d EMA50 + Tenkan > Kijun (bullish) + volume spike
            long_breakout = close[i] > cloud_top
            bullish_momentum = tenkan_aligned[i] > kijun_aligned[i]
            long_signal = long_breakout and price_above_ema and bullish_momentum and volume_spike[i]
            
            # Short: price breaks below cloud + price below 1d EMA50 + Tenkan < Kijun (bearish) + volume spike
            short_breakout = close[i] < cloud_bottom
            bearish_momentum = tenkan_aligned[i] < kijun_aligned[i]
            short_signal = short_breakout and price_below_ema and bearish_momentum and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches cloud bottom OR trend turns bearish (price below EMA) OR Tenkan < Kijun
            if (close[i] < cloud_bottom or not price_above_ema or tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches cloud top OR trend turns bullish (price above EMA) OR Tenkan > Kijun
            if (close[i] > cloud_top or not price_below_ema or tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0