#!/usr/bin/env python3
# 6h_1d_Ichimoku_Cloud_Breakout
# Hypothesis: Use 1d Ichimoku cloud (from daily timeframe) to determine trend and support/resistance on 6h chart.
# Enter long when 6h price breaks above the 1d cloud with price > Senkou Span A and B (bullish cloud).
# Enter short when 6h price breaks below the 1d cloud with price < Senkou Span A and B (bearish cloud).
# Use 6h volume confirmation (2x average volume) to avoid false breakouts.
# Works in bull/bear: cloud acts as dynamic support/resistance, trend filter inherent in cloud position.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Cloud_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 periods for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max()
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Shift Senkou Spans forward by 26 periods (for cloud plotting)
    # But for signal generation, we use current cloud values (already shifted in calculation)
    # Actually, we need to use the values as they would be known at time t (not forward-shifted)
    # So we use the calculated Senkou Spans without additional shift for current cloud
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # 6h volume confirmation
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Senkou Span B calculation
        # Get values
        close_val = prices['close'].iloc[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        span_a_val = senkou_span_a_aligned[i]
        span_b_val = senkou_span_b_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(span_a_val) or 
            np.isnan(span_b_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B form the cloud)
        upper_cloud = max(span_a_val, span_b_val)
        lower_cloud = min(span_a_val, span_b_val)
        
        if position == 0:
            # Long: Price breaks above cloud with volume confirmation
            if (close_val > upper_cloud and  # Price above cloud
                close_val > tenkan_val and   # Price above conversion line (momentum)
                vol_ratio_val > 2.0):      # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud with volume confirmation
            elif (close_val < lower_cloud and  # Price below cloud
                  close_val < tenkan_val and   # Price below conversion line (momentum)
                  vol_ratio_val > 2.0):      # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price closes below cloud or Tenkan-Kijun cross down
            if close_val < lower_cloud or (tenkan_val < kijun_val and close_val < kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price closes above cloud or Tenkan-Kijun cross up
            if close_val > upper_cloud or (tenkan_val > kijun_val and close_val > kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals