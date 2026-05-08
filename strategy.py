#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Ichimoku, trend filter, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Daily trend filter: price above/below Kumo (cloud)
    # Bullish: price above cloud (both spans)
    # Bearish: price below cloud (both spans)
    # Note: Cloud is shifted forward 26 periods, but we use current values for simplicity
    # as the alignment function handles the timing
    price_above_cloud = (close_1d > senkou_span_a) & (close_1d > senkou_span_b)
    price_below_cloud = (close_1d < senkou_span_a) & (close_1d < senkou_span_b)
    price_above_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_above_cloud.astype(float))
    price_below_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_below_cloud.astype(float))
    
    # Daily volume spike: current volume > 2.0 * 20-day average
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma20d * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Ichimoku (52 periods needed)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(price_above_cloud_aligned[i]) or np.isnan(price_below_cloud_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TK cross bullish (Tenkan > Kijun) + price above cloud + volume spike
            tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            long_cond = tk_cross_bullish and (price_above_cloud_aligned[i] > 0.5) and vol_spike_aligned[i]
            
            # Short entry: TK cross bearish (Tenkan < Kijun) + price below cloud + volume spike
            tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
            short_cond = tk_cross_bearish and (price_below_cloud_aligned[i] > 0.5) and vol_spike_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross bearish OR price falls below cloud
            tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
            price_below_cloud_now = price_below_cloud_aligned[i] > 0.5
            if tk_cross_bearish or price_below_cloud_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross bullish OR price rises above cloud
            tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            price_above_cloud_now = price_above_cloud_aligned[i] > 0.5
            if tk_cross_bullish or price_above_cloud_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku TK cross with cloud filter on daily timeframe, executed on 6H.
# The Ichimoku cloud acts as dynamic support/resistance, while TK cross signals momentum.
# Volume spike (2x 20-day average) confirms institutional participation.
# Works in bull markets (trend continuation above cloud) and bear markets (trend continuation below cloud).
# Cloud filter prevents whipsaws in sideways markets. Target: 50-150 total trades over 4 years.