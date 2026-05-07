#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Weekly trend filter: price above/below weekly cloud
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): current close shifted 26 periods behind
    # Not used for filtering in this strategy
    
    # Align Ichimoku components to 6h timeframe (only completed weekly values)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Daily volume spike detection (24-period average on 6s)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 52)  # Wait for volume MA and weekly Ichimoku
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or \
           np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is above or below weekly cloud
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, volume spike
            tk_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and \
                       tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            vol_condition = volume[i] > vol_ma[i] * 1.5
            
            if tk_cross and price_above_cloud and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and \
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and \
                  close[i] < cloud_bottom and vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Tenkan crosses below Kijun or price drops below cloud
            tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and \
                            tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
            price_below_cloud = close[i] < cloud_bottom
            
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses above Kijun or price rises above cloud
            tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and \
                          tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku TK cross with weekly cloud filter and volume confirmation.
# Weekly Ichimoku cloud provides dynamic support/resistance and trend direction.
# TK cross (Tenkan/Kijun crossover) signals momentum shifts.
# Volume confirmation ensures institutional participation in the breakout.
# Works in bull markets (buy when TK crosses above Kijun in uptrend/cloud) and 
# bear markets (sell when TK crosses below Kijun in downtrend/cloud).
# Position size 0.25 balances risk and keeps trade frequency moderate (~10-25 trades/year).