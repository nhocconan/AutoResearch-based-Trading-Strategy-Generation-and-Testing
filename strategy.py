#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
- Long: Tenkan > Kijun AND price > Senkou Span A AND price > Senkou Span B (bullish cloud) AND volume > 1.5x 24-period avg
- Short: Tenkan < Kijun AND price < Senkou Span A AND price < Senkou Span B (bearish cloud) AND volume > 1.5x 24-period avg
- Exit: Opposite Ichimoku cross OR price crosses cloud midpoint
- Uses 1d HTF for Ichimoku calculation (more stable on higher timeframe)
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Ichimoku provides trend, support/resistance, and momentum in one indicator
- Works in bull (buy above cloud) and bear (sell below cloud) markets
"""

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
    
    # Volume confirmation: > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d Ichimoku components (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters: 9, 26, 52
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (use prior completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 24)  # Need 52 for Senkou B, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Ichimoku signals
        bullish_cross = tenkan_aligned[i] > kijun_aligned[i]
        bearish_cross = tenkan_aligned[i] < kijun_aligned[i]
        price_above_cloud = (close[i] > senkou_a_aligned[i]) and (close[i] > senkou_b_aligned[i])
        price_below_cloud = (close[i] < senkou_a_aligned[i]) and (close[i] < senkou_b_aligned[i])
        
        if position == 0:
            # Long: Bullish TK cross AND price above cloud AND volume confirmation
            if bullish_cross and price_above_cloud and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross AND price below cloud AND volume confirmation
            elif bearish_cross and price_below_cloud and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish TK cross OR price below cloud
            if bearish_cross or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish TK cross OR price above cloud
            if bullish_cross or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1d_VolumeConfirm"
timeframe = "6h"
leverage = 1.0