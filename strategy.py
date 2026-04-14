#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with daily trend filter and volume confirmation.
# Uses Ichimoku (Tenkan-sen/Kijun-sen cross) for entry timing, daily Cloud (Senkou Span A/B) for trend filter,
# and volume > 1.5x average for confirmation. Works in bull/bear by only trading in direction of daily cloud.
# Targets 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Ichimoku components and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    if len(df_1d) < period_tenkan:
        return np.zeros(n)
    tenkan_sen = (pd.Series(df_1d['high']).rolling(window=period_tenkan, min_periods=period_tenkan).max() +
                  pd.Series(df_1d['low']).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    if len(df_1d) < period_kijun:
        return np.zeros(n)
    kijun_sen = (pd.Series(df_1d['high']).rolling(window=period_kijun, min_periods=period_kijun).max() +
                 pd.Series(df_1d['low']).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    if len(df_1d) < period_senkou_b:
        return np.zeros(n)
    senkou_span_b = (pd.Series(df_1d['high']).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() +
                     pd.Series(df_1d['low']).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe (with proper delay for completed daily bars)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Daily trend filter: price above/both spans = bullish, below/both = bearish
    # Bullish when price > max(Span A, Span B), Bearish when price < min(Span A, Span B)
    span_a_b = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    span_b_a = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(period_kijun, period_senkou_b, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(span_a_b[i]) or
            np.isnan(span_b_a[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signal: Tenkan-sen cross above/below Kijun-sen
        tenkan_above_kijun = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tenkan_below_kijun = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Trend filter: price relative to Kumo (cloud)
        price_above_cloud = close[i] > span_a_b[i]
        price_below_cloud = close[i] < span_b_a[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Tenkan crosses above Kijun + price above cloud + volume
            if (tenkan_above_kijun and 
                price_above_cloud and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Tenkan crosses below Kijun + price below cloud + volume
            elif (tenkan_below_kijun and 
                  price_below_cloud and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun or price enters cloud
            if (tenkan_below_kijun or 
                not price_above_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun or price enters cloud
            if (tenkan_above_kijun or 
                not price_below_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Ichimoku_Cloud_Volume_v2"
timeframe = "6h"
leverage = 1.0