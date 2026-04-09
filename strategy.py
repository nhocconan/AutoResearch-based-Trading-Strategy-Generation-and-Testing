#!/usr/bin/env python3
# 1d_weekly_ichimoku_trend_v1
# Hypothesis: Weekly Ichimoku Cloud on 1d timeframe for trend direction with volume confirmation.
# Long when price is above weekly Ichimoku Cloud with volume > 1.5x 20-period average.
# Short when price is below weekly Ichimoku Cloud with volume > 1.5x 20-period average.
# Exit when price crosses the weekly Tenkan-sen or Kijun-sen in opposite direction.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong trends in both bull and bear markets while avoiding sideways chop.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_ichimoku_trend_v1"
timeframe = "1d"
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
    
    # Get weekly data for Ichimoku Cloud (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for Ichimoku calculation
        return np.zeros(n)
    
    # Calculate weekly Ichimoku Cloud components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = close_1w  # Will be aligned properly with delay
    
    # Align Ichimoku components to daily timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b, additional_delay_bars=26)
    chikou_aligned = align_htf_to_ltf(prices, df_1w, chikou_span, additional_delay_bars=26)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: Price closes below Kijun-sen or below cloud (trend weakening)
            if close[i] < kijun_aligned[i] or close[i] < lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes above Kijun-sen or above cloud (trend weakening)
            if close[i] > kijun_aligned[i] or close[i] > upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry with volume confirmation
            bullish_setup = (close[i] > upper_cloud) and volume_confirmed
            bearish_setup = (close[i] < lower_cloud) and volume_confirmed
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals