#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation
# Long when Tenkan-sen > Kijun-sen, price above Kumo cloud, price > weekly EMA20, and volume > 1.5x 20-period average
# Short when Tenkan-sen < Kijun-sen, price below Kumo cloud, price < weekly EMA20, and volume > 1.5x 20-period average
# Exit when Tenkan-sen crosses back below/above Kijun-sen (opposite signal)
# Ichimoku provides multi-line trend confirmation; weekly EMA filters higher timeframe trend; volume confirms momentum
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (52 for Senkou Span B + buffer)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        if position == 0:
            # Long setup: Tenkan > Kijun, price above cloud, price > weekly EMA20, volume confirmation
            if (tenkan_sen[i] > kijun_sen[i] and 
                price > cloud_top and 
                price > ema20_1w_aligned[i] and 
                vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Tenkan < Kijun, price below cloud, price < weekly EMA20, volume confirmation
            elif (tenkan_sen[i] < kijun_sen[i] and 
                  price < cloud_bottom and 
                  price < ema20_1w_aligned[i] and 
                  vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Tenkan crosses back below Kijun (opposite signal)
            if tenkan_sen[i] < kijun_sen[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Tenkan crosses back above Kijun (opposite signal)
            if tenkan_sen[i] > kijun_sen[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_1wEMA20_Volume"
timeframe = "6h"
leverage = 1.0