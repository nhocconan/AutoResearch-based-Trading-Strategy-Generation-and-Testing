#!/usr/bin/env python3
"""
6h_1D_Ichimoku_Cloud_Breakout_With_Volume_Filter
Hypothesis: 6h price breaks above/below daily Ichimoku Cloud (Senkou Span A/B) with 6h volume > 1.8x 20-period average.
Long when price breaks above Senkou Span A + volume condition.
Short when price breaks below Senkou Span B + volume condition.
Exit when price crosses Tenkan-sen/Kijun-sen crossover in opposite direction.
Designed for 6h timeframe with 1d Ichimoku for trend context. Works in bull/bear via trend filter.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h (with proper forward shift for Senkou spans)
    # Senkou spans are already shifted 26 periods ahead in their calculation
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6h volume condition: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_condition = volume > (vol_ma_20 * 1.8)
    
    # Tenkan/Kijun crossover for trend and exit signals
    tk_cross = tenkan_sen - kijun_sen  # Positive when Tenkan > Kijun (bullish)
    tk_cross_aligned = align_htf_to_ltf(prices, df_1d, tk_cross)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tk_cross_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B form the cloud)
        # In Ichimoku, the cloud is between Senkou Span A and Senkou Span B
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Bullish when Senkou Span A > Senkou Span B (cloud is bullish)
        bullish_cloud = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        bearish_cloud = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = close[i] > upper_cloud and vol_condition[i] and bullish_cloud
        short_breakout = close[i] < lower_cloud and vol_condition[i] and bearish_cloud
        
        # Exit conditions: Tenkan/Kijun cross in opposite direction
        long_exit = tk_cross_aligned[i] < 0  # Tenkan crossed below Kijun
        short_exit = tk_cross_aligned[i] > 0  # Tenkan crossed above Kijun
        
        if position == 0:
            if long_breakout:
                position = 1
                signals[i] = position_size
            elif short_breakout:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1D_Ichimoku_Cloud_Breakout_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0