#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with Tenkan/Kijun cross and daily trend filter
# Uses Kumo (cloud) as dynamic support/resistance and TK cross for momentum.
# Daily trend filter (price vs Kumo) ensures alignment with higher timeframe bias.
# Works in bull/bear by only taking trades in direction of daily Kumo twist.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line)
    tenkan_sen = np.full(len(df_1d), np.nan)
    for i in range(tenkan_period - 1, len(df_1d)):
        period_high = np.max(high_1d[i - tenkan_period + 1:i + 1])
        period_low = np.min(low_1d[i - tenkan_period + 1:i + 1])
        tenkan_sen[i] = (period_high + period_low) / 2
    
    # Calculate Kijun-sen (Base Line)
    kijun_sen = np.full(len(df_1d), np.nan)
    for i in range(kijun_period - 1, len(df_1d)):
        period_high = np.max(high_1d[i - kijun_period + 1:i + 1])
        period_low = np.min(low_1d[i - kijun_period + 1:i + 1])
        kijun_sen[i] = (period_high + period_low) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B)
    senkou_span_b = np.full(len(df_1d), np.nan)
    for i in range(senkou_span_b_period - 1, len(df_1d)):
        period_high = np.max(high_1d[i - senkou_span_b_period + 1:i + 1])
        period_low = np.min(low_1d[i - senkou_span_b_period + 1:i + 1])
        senkou_span_b[i] = (period_high + period_low) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for 1d close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Daily trend filter: price above/below Kumo (cloud)
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Kumo twist detection: Senkou Span A crossing Senkou Span B
    kumo_twist_bullish = senkou_span_a_aligned > senkou_span_b_aligned  # A above B = bullish twist
    kumo_twist_bearish = senkou_span_a_aligned < senkou_span_b_aligned  # A below B = bearish twist
    
    # Volume filter: volume > 1.5 x 24-period average (4 days of 6h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Ichimoku components (52), volume MA (24)
    start_idx = max(52, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # TK cross signals
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Price vs cloud
        price_above_cloud = price > cloud_top[i]
        price_below_cloud = price < cloud_bottom[i]
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + bullish Kumo twist + volume
            if (tk_cross_bullish and price_above_cloud and 
                kumo_twist_bullish[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: TK cross bearish + price below cloud + bearish Kumo twist + volume
            elif (tk_cross_bearish and price_below_cloud and 
                  kumo_twist_bearish[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross bearish or price drops below cloud
            if tk_cross_bearish or price < cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross bullish or price rises above cloud
            if tk_cross_bullish or price > cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dKumoTwist_Volume"
timeframe = "6h"
leverage = 1.0