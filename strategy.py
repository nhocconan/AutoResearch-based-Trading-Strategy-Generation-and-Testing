# 6h_1d_Ichimoku_Kumo_Twist_VolumeFilter
# Hypothesis: Ichimoku cloud twist on 1d (Tenkan/Kijun cross with cloud color change) 
# indicates strong momentum shift. Enter on 6h when price breaks cloud in direction of twist,
# with volume confirmation. Works in both bull/bear as it captures momentum shifts.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.

name = "6h_1d_Ichimoku_Kumo_Twist_VolumeFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku (26*2)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Cloud color: green when Span A > Span B (bullish), red when Span A < Span B (bearish)
    # Cloud twist occurs when Senkou Span A and Senkou Span B cross
    span_a = senkou_span_a.values
    span_b = senkou_span_b.values
    
    # Detect cloud twist: Span A crosses above/below Span B
    # Bullish twist: Span A crosses above Span B (previous Span A <= Span B, current Span A > Span B)
    # Bearish twist: Span A crosses below Span B (previous Span A >= Span B, current Span A < Span B)
    bullish_twist = (np.roll(span_a, 1) <= np.roll(span_b, 1)) & (span_a > span_b)
    bearish_twist = (np.roll(span_a, 1) >= np.roll(span_b, 1)) & (span_a < span_b)
    
    # Align 1d indicators to 6h timeframe
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    span_a_aligned = align_htf_to_ltf(prices, df_1d, span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, span_b)
    
    # Calculate volume ratio (current vs 20-period average) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_ratio = volume / (vol_ma.values + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure Ichimoku is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or 
            np.isnan(bullish_twist_aligned[i]) or np.isnan(bearish_twist_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: need above average volume
        vol_confirmed = volume_ratio[i] > 1.2
        
        if position == 0:
            # Look for cloud twist with price breaking cloud in direction of twist
            # Bullish setup: bullish twist + price above cloud (Span A and Span B)
            if bullish_twist_aligned[i] and vol_confirmed:
                if close[i] > span_a_aligned[i] and close[i] > span_b_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Bearish setup: bearish twist + price below cloud
            elif bearish_twist_aligned[i] and vol_confirmed:
                if close[i] < span_a_aligned[i] and close[i] < span_b_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price closes back into cloud or opposite twist
            if close[i] < span_a_aligned[i] or close[i] < span_b_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif bearish_twist_aligned[i]:  # Opposite twist signals trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes back into cloud or opposite twist
            if close[i] > span_a_aligned[i] or close[i] > span_b_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif bullish_twist_aligned[i]:  # Opposite twist signals trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals