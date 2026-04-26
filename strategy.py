#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, use Ichimoku cloud twist (Senkou Span A/B cross) from 1d as trend change signal, confirmed by price breaking above/below cloud with volume spike (>1.5x 20-period average). Enter long when Senkou Span A crosses above Senkou Span B (bullish twist) and price > Senkou Span A with volume spike. Enter short when Senkou Span A crosses below Senkou Span B (bearish twist) and price < Senkou Span B with volume spike. Uses discrete position size 0.25. Designed for 12-30 trades/year on 6h by requiring Ichimoku twist (rare event) and volume confirmation, reducing overtrading while capturing major trend changes in both bull and bear markets.
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    # Note: Senkou spans are already shifted 26 periods ahead in calculation,
    # so we align the calculated values (which represent future cloud)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52-period for Senkou B, 20 for volume MA
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku twist detection (Senkou A/B cross)
        # Bullish twist: Senkou A crosses above Senkou B
        bullish_twist = (senkou_a_aligned[i] > senkou_b_aligned[i]) and \
                       (i > start_idx and senkou_a_aligned[i-1] <= senkou_b_aligned[i-1])
        # Bearish twist: Senkou A crosses below Senkou B
        bearish_twist = (senkou_a_aligned[i] < senkou_b_aligned[i]) and \
                       (i > start_idx and senkou_a_aligned[i-1] >= senkou_b_aligned[i-1])
        
        if position == 0:
            # Long: bullish twist + price > Senkou A + volume spike
            long_signal = bullish_twist and (close[i] > senkou_a_aligned[i]) and volume_spike[i]
            
            # Short: bearish twist + price < Senkou B + volume spike
            short_signal = bearish_twist and (close[i] < senkou_b_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < Senkou A OR bearish twist occurs
            if (close[i] < senkou_a_aligned[i] or bearish_twist):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > Senkou B OR bullish twist occurs
            if (close[i] > senkou_b_aligned[i] or bullish_twist):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0