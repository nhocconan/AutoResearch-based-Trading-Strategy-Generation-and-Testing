#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Kumo cloud (Senkou Span A/B) and Tenkan > Kijun in 1d uptrend (close > EMA50) with volume > 1.5x 20-bar avg
# - Short when price breaks below Kumo cloud and Tenkan < Kijun in 1d downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20 trades/year (80 total over 4 years) to avoid fee drag
# - 1d trend filter ensures alignment with higher timeframe momentum
# - Ichimoku provides dynamic support/resistance and trend confirmation
# - Volume confirmation filters low-participation breakouts

name = "6h_1d_ichimoku_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = prices['high'].rolling(window=9, min_periods=9).max().values
    period9_low = prices['low'].rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = prices['high'].rolling(window=26, min_periods=26).max().values
    period26_low = prices['low'].rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = prices['high'].rolling(window=52, min_periods=52).max().values
    period52_low = prices['low'].rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo cloud boundaries: max/min of Senkou Span A/B
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price above cloud, Tenkan > Kijun, 1d uptrend, volume spike
            if (prices['close'].iloc[i] > upper_cloud[i] and 
                tenkan_sen[i] > kijun_sen[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price below cloud, Tenkan < Kijun, 1d downtrend, volume spike
            elif (prices['close'].iloc[i] < lower_cloud[i] and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price re-enters the cloud (cloud acts as dynamic support/resistance)
            if position == 1 and prices['close'].iloc[i] < upper_cloud[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > lower_cloud[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals