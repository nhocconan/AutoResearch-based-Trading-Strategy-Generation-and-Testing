#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_trend_with_volume
# Hypothesis: 6-hour Ichimoku cloud trend following with 1d cloud filter and volume confirmation.
# Uses Ichimoku (Tenkan, Kijun, Senkou) to identify trend direction and cloud for support/resistance.
# Volume confirmation reduces false breakouts. Works in bull/bear by following major trends.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "6h_1d_ichimoku_cloud_trend_with_volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_high = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    tenkan_low = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_high = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max()
    kijun_low = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_high_b = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    senkou_low_b = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = (senkou_high_b + senkou_low_b) / 2
    
    # Chikou Span (Lagging Span): not used for signals
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Bullish: price above cloud, Tenkan > Kijun, volume confirmation
        bullish = (close[i] > cloud_top and 
                   tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                   vol_confirm[i])
        
        # Bearish: price below cloud, Tenkan < Kijun, volume confirmation
        bearish = (close[i] < cloud_bottom and 
                   tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                   vol_confirm[i])
        
        # Enter long on bullish signal
        if bullish and position != 1:
            position = 1
            signals[i] = 0.25
        # Enter short on bearish signal
        elif bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses back into cloud or opposite Tenkan/Kijun cross
        elif position == 1 and (close[i] < cloud_bottom or tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > cloud_top or tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals