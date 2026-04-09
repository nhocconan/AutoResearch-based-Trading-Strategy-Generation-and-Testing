#!/usr/bin/env python3
# 6h_ichimoku_cloud_breakout_volume_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d timeframe with volume confirmation.
# In both bull and bear markets, price tends to respect the Ichimoku cloud as dynamic support/resistance.
# Breakouts above/below the cloud with volume confirmation capture strong momentum moves.
# The cloud acts as a filter: only go long when price is above cloud (bullish bias) or short when below cloud (bearish bias).
# Volume confirmation reduces false breakouts. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years by requiring cloud breakout + volume spike.
# Primary timeframe: 6h, HTF: 1d for Ichimoku calculation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_breakout_volume_v1"
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
    
    # 1d HTF data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                      pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou_span = pd.Series(close_1d).shift(-displacement)
    
    # Align Ichimoku components to 6h timeframe
    # Note: We use the completed 1d bar's values, so no additional displacement needed for alignment
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price moves back below cloud or volume dries up
            if close[i] < cloud_bottom or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves back above cloud or volume dries up
            if close[i] > cloud_top or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above cloud with volume confirmation
                if close[i] > cloud_top and high[i] > cloud_top:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below cloud with volume confirmation
                elif close[i] < cloud_bottom and low[i] < cloud_bottom:
                    position = -1
                    signals[i] = -0.25
    
    return signals