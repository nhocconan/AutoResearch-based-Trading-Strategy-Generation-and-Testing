#!/usr/bin/env python3
"""
12h_Williams_Alligator_Trend_Filter
Strategy: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d volume confirmation.
Long: Lips > Teeth > Jaw (bullish alignment) + volume > 1.3x 24-period average
Short: Lips < Teeth < Jaw (bearish alignment) + volume > 1.3x 24-period average
Exit: Alignment breaks
Position size: 0.25
Designed to catch strong trends while avoiding choppy markets.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator components (13,8,5 smoothed)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # SMMA (Smoothed Moving Average) function
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma((high + low) / 2, jaw_period)  # Jaw: 13-period SMMA of median price
    teeth = smma((high + low) / 2, teeth_period)  # Teeth: 8-period SMMA
    lips = smma((high + low) / 2, lips_period)  # Lips: 5-period SMMA
    
    # Get 1d volume average (24-period = 12 days of 12h data)
    volume_ma24 = np.convolve(volume, np.ones(24)/24, mode='full')[:len(volume)]
    volume_ma24 = np.concatenate([np.full(23, np.nan), volume_ma24[23:]])
    
    # Get 1d trend filter (close > open = uptrend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    trend_1d = (df_1d['close'] > df_1d['open']).astype(float).values  # 1 for up, 0 for down
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from sufficient warmup
    start_idx = max(jaw_period, teeth_period, lips_period, 24)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_ma24[i]) or np.isnan(trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current volume
        volume_current = volume[i]
        volume_filter = volume_current > (1.3 * volume_ma24[i])
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Entry signals
        if position == 0:
            # Long: Bullish alignment + volume filter + 1d uptrend
            if bullish_alignment and volume_filter and trend_1d_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume filter + 1d downtrend
            elif bearish_alignment and volume_filter and trend_1d_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bullish alignment breaks
            if not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bearish alignment breaks
            if not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Trend_Filter"
timeframe = "12h"
leverage = 1.0