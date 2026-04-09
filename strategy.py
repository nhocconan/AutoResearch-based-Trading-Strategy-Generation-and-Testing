#!/usr/bin/env python3
# 6h_ichimoku_trend_regime_v1
# Hypothesis: 6h strategy using Ichimoku Cloud from 1d timeframe as trend filter.
# Long when: price > 6h 20-period EMA AND price > 1d Ichimoku Cloud (bullish regime)
# Short when: price < 6h 20-period EMA AND price < 1d Ichimoku Cloud (bearish regime)
# Exit when price crosses back below/above the 6h 20-period EMA.
# Uses Ichimoku Cloud (Senkou Span A/B) from daily chart to define major trend regime.
# Designed to avoid counter-trend trades in strong trends and capture continuation moves.
# Target: 12-30 trades/year (50-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_trend_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h 20-period EMA for entry timing
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Get daily data for Ichimoku Cloud (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Align Ichimoku components to 6h timeframe (default delay = 1 completed bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # The Cloud boundaries: Senkou Span A and B
    # For bullish cloud: Senkou Span A > Senkou Span B
    # For bearish cloud: Senkou Span A < Senkou Span B
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_20[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses back below 6h 20-period EMA
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses back above 6h 20-period EMA
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for regime alignment
            bullish_regime = close[i] > cloud_top[i]  # Price above cloud = bullish regime
            bearish_regime = close[i] < cloud_bottom[i]  # Price below cloud = bearish regime
            
            # Only trade in alignment with major trend regime
            if bullish_regime and close[i] > ema_20[i]:
                position = 1
                signals[i] = 0.25
            elif bearish_regime and close[i] < ema_20[i]:
                position = -1
                signals[i] = -0.25
    
    return signals