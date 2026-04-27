#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_Filter
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 6h with 1w trend filter (price >/<
1w Kumo center) and volume confirmation. Works in bull/bear: In strong trends (price above/below
1w Kumo center), cloud twists signal momentum shifts. Volume spike confirms institutional interest.
Exit on opposite cloud twist or trend reversal (price crosses 1w Kumo center).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w Kumo (Ichimoku cloud) center: (Senkou Span A + Senkou Span B) / 2
    # Senkou Span A = (Tenkan-sen + Kijun-sen) / 2
    # Senkou Span B = (52-period high + 52-period low) / 2
    # Tenkan-sen = (9-period high + 9-period low) / 2
    # Kijun-sen = (26-period high + 26-period low) / 2
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (9-period)
    tenkan_sen = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (26-period)
    kijun_sen = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (52-period)
    senkou_span_b = (pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Kumo center
    kumo_center = (senkou_span_a + senkou_span_b) / 2
    kumo_center_aligned = align_htf_to_ltf(prices, df_1w, kumo_center)
    
    # 6h Ichimoku for cloud twist signal
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (9-period) on 6h
    tenkan_6h = (pd.Series(high_6h).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_6h).rolling(window=9, min_periods=9).min()) / 2
    tenkan_6h = tenkan_6h.values
    
    # Kijun-sen (26-period) on 6h
    kijun_6h = (pd.Series(high_6h).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_6h).rolling(window=26, min_periods=26).min()) / 2
    kijun_6h = kijun_6h.values
    
    # Senkou Span A on 6h (shifted 26 periods ahead)
    senkou_a_6h = (tenkan_6h + kijun_6h) / 2
    
    # Senkou Span B on 6h (52-period, shifted 26 periods ahead)
    senkou_b_6h = (pd.Series(high_6h).rolling(window=52, min_periods=52).max() + 
                   pd.Series(low_6h).rolling(window=52, min_periods=52).min()) / 2
    senkou_b_6h = senkou_b_6h.values
    
    # Cloud twist: Senkou A crosses Senkou B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    bullish_twist = (senkou_a_6h > senkou_b_6h) & (np.roll(senkou_a_6h, 1) <= np.roll(senkou_b_6h, 1))
    bearish_twist = (senkou_a_6h < senkou_b_6h) & (np.roll(senkou_a_6h, 1) >= np.roll(senkou_b_6h, 1))
    
    # Align twist signals to current bar (no look-ahead: twist confirmed at close)
    bullish_twist_aligned = align_htf_to_ltf(prices, prices, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, prices, bearish_twist.astype(float))
    
    # Volume spike: current > 1.8 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 1w data (52 periods) and 6h Ichimoku (52 periods)
    start_idx = max(60, 52)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumo_center_aligned[i]) or 
            np.isnan(bullish_twist_aligned[i]) or 
            np.isnan(bearish_twist_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kumo_val = kumo_center_aligned[i]
        bull_twist = bullish_twist_aligned[i] > 0.5
        bear_twist = bearish_twist_aligned[i] > 0.5
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Ichimoku cloud twist with 1w trend alignment and volume spike
            long_condition = (bull_twist and 
                             close_val > kumo_val and 
                             vol_spike)
            short_condition = (bear_twist and 
                              close_val < kumo_val and 
                              vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif bear_twist and close_val < kumo_val and vol_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: bearish twist or price crosses below 1w Kumo center (trend reversal)
            if bear_twist or close_val < kumo_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish twist or price crosses above 1w Kumo center (trend reversal)
            if bull_twist or close_val > kumo_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0