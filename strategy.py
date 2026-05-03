#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku TK Cross with 1d Cloud Filter and Volume Confirmation
# Uses Ichimoku Tenkan/Kijun cross for momentum signals, filtered by 1d Kumo (cloud) direction
# to ensure alignment with higher timeframe trend. Volume confirmation filters false breakouts.
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
# Works in both bull and bear markets by taking signals only when price is above/below cloud.

name = "6h_IchimokuTK_Cross_1dCloud_Volume"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values +
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values +
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values +
                pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 1d volume spike (volume > 1.8 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.8 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after sufficient warmup for Ichimoku
        # Skip if any value is NaN or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1] and
                close[i] > cloud_top and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1] and
                  close[i] < cloud_bottom and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish or price drops below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] or close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish or price rises above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] or close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals