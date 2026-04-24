#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d Williams %R filter and volume confirmation.
- Uses 4h timeframe (primary) and 1d HTF for Williams %R regime filter
- Camarilla levels calculated from prior 1d OHLC: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
- Breakout logic: long when price crosses above H3 with Williams %R < -80 (oversold) and volume confirmation
  short when price crosses below L3 with Williams %R > -20 (overbought) and volume confirmation
- Volume confirmation: current volume > 1.8 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to prior 1d close (mean reversion)
- Discrete signal size: 0.25 to balance return and risk
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Williams %R filter ensures we buy oversold bounces and sell overbought rejections, working in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R for regime filter (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate prior 1d Camarilla levels (H3 and L3)
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need Williams %R(14) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above H3 AND Williams %R < -80 (oversold) AND volume confirmation
            if (close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1] and 
                williams_r_aligned[i] < -80 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L3 AND Williams %R > -20 (overbought) AND volume confirmation
            elif (close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1] and 
                  williams_r_aligned[i] > -20 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 1d close (mean reversion) or reverse signal
            prior_close_1d = df_1d['close'].shift(1).values
            prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close_1d)
            if close[i] <= prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 1d close (mean reversion) or reverse signal
            prior_close_1d = df_1d['close'].shift(1).values
            prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close_1d)
            if close[i] >= prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dWilliamsR_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0