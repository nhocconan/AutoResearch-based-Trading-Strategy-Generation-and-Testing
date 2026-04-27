#!/usr/bin/env python3
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
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full(len(high_1d), np.nan)
    period9_low = np.full(len(low_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= 8:
            period9_high[i] = np.max(high_1d[i-8:i+1])
            period9_low[i] = np.min(low_1d[i-8:i+1])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full(len(high_1d), np.nan)
    period26_low = np.full(len(low_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= 25:
            period26_high[i] = np.max(high_1d[i-25:i+1])
            period26_low[i] = np.min(low_1d[i-25:i+1])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full(len(high_1d), np.nan)
    period52_low = np.full(len(low_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= 51:
            period52_high[i] = np.max(high_1d[i-51:i+1])
            period52_low[i] = np.min(low_1d[i-51:i+1])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get weekly data for trend filter: EMA(50) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i < 49:
            ema_1w_50[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_50[i-1]):
                ema_1w_50[i] = np.mean(close_1w[i-49:i+1])
            else:
                ema_1w_50[i] = close_1w[i] * alpha_w + ema_1w_50[i-1] * (1 - alpha_w)
    
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate volume ratio: current volume / 24-period average volume
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_ma_24)) & (vol_ma_24 > 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma_24[valid_vol]
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(52, 50, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_1w_50_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine Kumo (cloud) boundaries
        senkou_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        senkou_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Check if price is above or below cloud
        price_above_cloud = price > senkou_top
        price_below_cloud = price < senkou_bottom
        
        # TK Cross signals
        tk_cross_bull = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_bear = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        # Volume filter: require volume expansion
        volume_expansion = volume_ratio[i] > 1.5
        
        if position == 0:
            # Long: TK bullish cross + price above cloud + weekly uptrend + volume expansion
            if (tk_cross_bull and 
                price_above_cloud and 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1] and
                volume_expansion):
                signals[i] = 0.25
                position = 1
            # Short: TK bearish cross + price below cloud + weekly downtrend + volume expansion
            elif (tk_cross_bear and 
                  price_below_cloud and 
                  ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1] and
                  volume_expansion):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: TK bearish cross or price drops below cloud or weekly trend turns down
            if (tk_cross_bear or 
                price_below_cloud or 
                ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK bullish cross or price rises above cloud or weekly trend turns up
            if (tk_cross_bull or 
                price_above_cloud or 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTK_Cross_CloudFilter_WeeklyEMA50_v1"
timeframe = "6h"
leverage = 1.0