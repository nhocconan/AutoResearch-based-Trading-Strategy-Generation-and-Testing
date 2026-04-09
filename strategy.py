#!/usr/bin/env python3
# 6h_ichimoku_trend_follow_v3
# Hypothesis: 6h Ichimoku trend following with 12h/1d confirmation. Long when price > Kumo cloud, Tenkan > Kijun, and Chikou Span above price 26 periods ago. Short when price < Kumo cloud, Tenkan < Kijun, and Chikou Span below price 26 periods ago. Uses 12h EMA50 as higher timeframe filter to avoid counter-trend trades. Discrete position sizing (0.25) to limit fee drag. Target: 12-37 trades/year. Works in bull/bear by following higher timeframe trend and requiring Ichimoku alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_trend_follow_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough for Ichimoku (52 periods) + warmup
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    chikou_shift = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    low_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    low_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    low_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou = np.roll(close, chikou_shift)  # Shifted back, so index i corresponds to close[i - chikou_shift]
    # First chikou_shift values will be invalid (from roll), handled by min_periods equivalent
    
    # Current Kumo (cloud) boundaries: Senkou Span A and B shifted 26 periods back to align with current price
    # We need values from 26 periods ago for current cloud
    senkou_span_a_lagged = np.roll(senkou_span_a, chikou_shift)
    senkou_span_b_lagged = np.roll(senkou_span_b, chikou_shift)
    # First chikou_shift values invalid
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    cloud_bottom = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    
    # 12h HTF trend filter: 50-period EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(chikou_shift, n):  # Start after Chikou shift
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(chikou[i]) or np.isnan(close[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku conditions
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        chikou_above_price = chikou[i] > close[i]
        chikou_below_price = chikou[i] < close[i]
        
        if position == 1:  # Long position
            # Exit: price falls below cloud OR Tenkan < Kijun (trend weakness)
            if close[i] <= cloud_top[i] or tenkan[i] < kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR Tenkan > Kijun (trend weakness)
            if close[i] >= cloud_bottom[i] or tenkan[i] > kijun[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with 12h trend alignment and full Ichimoku alignment
            # Bullish: price > cloud, Tenkan > Kijun, Chikou > price, and 12h EMA50 uptrend
            bullish_setup = (price_above_cloud and tenkan_above_kijun and chikou_above_price and 
                           close[i] > ema_50_12h_aligned[i])
            # Bearish: price < cloud, Tenkan < Kijun, Chikou < price, and 12h EMA50 downtrend
            bearish_setup = (price_below_cloud and tenkan_below_kijun and chikou_below_price and 
                           close[i] < ema_50_12h_aligned[i])
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals