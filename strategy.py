#!/usr/bin/env python3
"""
12h_Williams_Alligator_Trend_Filter_v1
Hypothesis: Uses Williams Alligator (Jaw/Teeth/Lips) on 12h timeframe to identify trend direction,
combined with 1-week trend filter and volume confirmation to avoid false signals.
Williams Alligator uses smoothed moving averages (SMMA) to filter noise and identify
trending vs ranging markets. Targets 15-30 trades/year to minimize fee drift.
Works in both bull and bear markets by following the trend direction from higher timeframes.
"""

name = "12h_Williams_Alligator_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 12h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # All are SMMA of median price (high+low)/2
    median_price = (high + low) / 2
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Align Alligator lines to lower timeframe (12h data already matches our timeframe)
    jaw_aligned = jaw  # Already on 12h timeframe
    teeth_aligned = teeth
    lips_aligned = lips
    
    # 1-week trend filter: EMA of weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        is_uptrend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        is_downtrend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: uptrend + price above lips + 1-week trend up + volume confirmation
            if (is_uptrend and close[i] > lips_aligned[i] and 
                close[i] > ema_1w_aligned[i] and volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price below lips + 1-week trend down + volume confirmation
            elif (is_downtrend and close[i] < lips_aligned[i] and 
                  close[i] < ema_1w_aligned[i] and volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend changes to downtrend OR price crosses below lips
            if not is_uptrend or close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend changes to uptrend OR price crosses above lips
            if not is_downtrend or close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals