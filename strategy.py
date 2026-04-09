#!/usr/bin/env python3
# 6h_ichimoku_trend_follow_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d timeframe for trend direction,
# with TK (Tenkan-Kijun) cross on 6h for entry timing and volume confirmation.
# Long: Price above 1d cloud, TK cross bullish on 6h, volume > 1.5x 20-period average.
# Short: Price below 1d cloud, TK cross bearish on 6h, volume > 1.5x 20-period average.
# Exit: TK cross reverses or price exits the cloud.
# Ichimoku provides strong trend filtering; TK cross gives timely entries.
# Works in both bull (trend follow) and bear (counter-trend via cloud breaks).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:  # Need at least 26 periods for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26  # Kumo cloud displacement
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_10 = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    low_10 = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = ((high_10 + low_10) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max()
    low_26 = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    low_52 = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_b = ((high_52 + low_52) / 2).values
    
    # Align Ichimoku components to 6h (displacement handled by align_htf_to_ltf)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get 6h data for TK cross
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # 6h Tenkan-sen and Kijun-sen
    high_9_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max()
    low_9_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min()
    tenkan_6h = ((high_9_6h + low_9_6h) / 2).values
    
    high_26_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max()
    low_26_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min()
    kijun_6h = ((high_26_6h + low_26_6h) / 2).values
    
    # TK cross signals: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
    tk_bullish = tenkan_6h > kijun_6h
    tk_bearish = tenkan_6h < kijun_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup (max period 52)
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries: Senkou Span A and B form the cloud
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: TK cross turns bearish OR price falls below cloud
            if not tk_bullish[i] or close[i] < cloud_bottom:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross turns bullish OR price rises above cloud
            if not tk_bearish[i] or close[i] > cloud_top:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price above cloud, TK bullish cross, volume confirmation
            if (close[i] > cloud_top and      # Price above cloud
                tk_bullish[i] and             # TK bullish
                volume_confirmed):            # Volume spike
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud, TK bearish cross, volume confirmation
            elif (close[i] < cloud_bottom and # Price below cloud
                  tk_bearish[i] and           # TK bearish
                  volume_confirmed):          # Volume spike
                position = -1
                signals[i] = -0.25
    
    return signals