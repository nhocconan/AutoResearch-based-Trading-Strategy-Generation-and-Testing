#!/usr/bin/env python3
# 6h_1d_ichimoku_trend_v1
# Hypothesis: 6h strategy using 1d Ichimoku cloud with TK cross and cloud filter.
# Long when Tenkan > Kijun and price above cloud (bullish alignment).
# Short when Tenkan < Kijun and price below cloud (bearish alignment).
# Uses volume confirmation to filter breakouts (volume > 1.5x 20-period average).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Ichimoku is proven in trending markets and works across regimes via cloud filter.
# Target: 12-30 trades/year (50-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_trend_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Ichimoku (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    low_9 = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max()
    low_26 = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    low_52 = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Determine cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price closes below cloud OR Tenkan < Kijun (trend change)
            if close[i] < cloud_bottom[i] or tenkan_aligned[i] < kijun_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes above cloud OR Tenkan > Kijun (trend change)
            if close[i] > cloud_top[i] or tenkan_aligned[i] > kijun_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Ichimoku signal with volume confirmation
            bullish_aligned = (tenkan_aligned[i] > kijun_aligned[i]) and (close[i] > cloud_top[i])
            bearish_aligned = (tenkan_aligned[i] < kijun_aligned[i]) and (close[i] < cloud_bottom[i])
            
            if bullish_aligned and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif bearish_aligned and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals