#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudTwist_v2
Hypothesis: Trade Ichimoku TK cross with 1d cloud twist filter on 6h timeframe. 
Enter long when TK crosses above AND price > 1d cloud (bullish twist). 
Enter short when TK crosses below AND price < 1d cloud (bearish twist). 
Exit on opposite TK cross or when price re-enters the cloud. 
Uses volume confirmation (>1.3x 20-period average) to reduce false signals. 
Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year. 
Works in bull (TK cross above cloud) and bear (TK cross below cloud) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need sufficient data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = close_1d  # Not used in signals but calculated for completeness
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and volume MA (20)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine Ichimoku cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK Cross conditions
        tk_cross_above = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_below = tenkan_aligned[i] < kijun_aligned[i]
        
        # Previous bar TK cross (to detect fresh crosses)
        prev_tk_cross_above = tenkan_aligned[i-1] > kijun_aligned[i-1]
        prev_tk_cross_below = tenkan_aligned[i-1] < kijun_aligned[i-1]
        
        # Fresh TK cross (happened on this bar)
        fresh_tk_cross_above = tk_cross_above and not prev_tk_cross_above
        fresh_tk_cross_below = tk_cross_below and not prev_tk_cross_below
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        price_in_cloud = (close[i] >= lower_cloud) and (close[i] <= upper_cloud)
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: fresh TK cross above + price above cloud + volume confirmation
            long_setup = fresh_tk_cross_above and price_above_cloud and volume_confirm
            
            # Short setup: fresh TK cross below + price below cloud + volume confirmation
            short_setup = fresh_tk_cross_below and price_below_cloud and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TK cross below OR price re-enters cloud
            if (fresh_tk_cross_below) or price_in_cloud:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TK cross above OR price re-enters cloud
            if (fresh_tk_cross_above) or price_in_cloud:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudTwist_v2"
timeframe = "6h"
leverage = 1.0