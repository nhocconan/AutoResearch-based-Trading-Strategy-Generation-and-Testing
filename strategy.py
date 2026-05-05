#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud Breakout with Weekly Trend Filter and Volume Confirmation
# Long when price breaks above weekly Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND volume spike
# Short when price breaks below weekly Kumo AND Tenkan < Kijun (bearish TK cross) AND volume spike
# Weekly Ichimoku provides strong trend filter (cloud = support/resistance, TK cross = momentum)
# Volume spike (2.0x 20-bar MA) confirms breakout validity
# Works in bull (cloud acts as support, breaks upward) and bear (cloud acts as resistance, breaks downward)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing high-probability breaks
# Timeframe: 6h (primary timeframe as required)

name = "6h_Ichimoku_WeeklyCloud_TKCross_VolumeSpike"
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
    
    # Get weekly data ONCE before loop for Ichimoku cloud and TK cross
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    if len(high_1w) >= period_tenkan:
        tenkan_sen = (pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values +
                      pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values) / 2.0
    else:
        tenkan_sen = np.full(len(high_1w), np.nan)
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    if len(high_1w) >= period_kijun:
        kijun_sen = (pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values +
                     pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values) / 2.0
    else:
        kijun_sen = np.full(len(high_1w), np.nan)
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    if len(high_1w) >= period_senkou_b:
        senkou_span_b = (pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values +
                         pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values) / 2.0
    else:
        senkou_span_b = np.full(len(high_1w), np.nan)
    
    # Chikou Span (Lagging Span): not used for signals (plotted 26 periods behind)
    
    # Align Ichimoku components to 6h timeframe (wait for weekly bar to close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Weekly Kumo (cloud) boundaries: max/min of Senkou Span A and B
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume confirmation on 6h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN (due to insufficient data for indicators)
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above cloud TOP AND bullish TK cross (Tenkan > Kijun) AND volume spike
            if (close[i] > cloud_top[i] and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud BOTTOM AND bearish TK cross (Tenkan < Kijun) AND volume spike
            elif (close[i] < cloud_bottom[i] and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below cloud BOTTOM OR TK cross turns bearish
            if close[i] < cloud_bottom[i] or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above cloud TOP OR TK cross turns bullish
            if close[i] > cloud_top[i] or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals