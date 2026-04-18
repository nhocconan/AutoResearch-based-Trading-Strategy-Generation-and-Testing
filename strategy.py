#!/usr/bin/env python3
"""
1d_WK1_WilliamsAlligator_Filtered_V1
Hypothesis: Use weekly Williams Alligator (jaw/teeth/lips) to define trend direction and market regime on 1d timeframe. 
Go long when price > Alligator Lips (green) and Lips > Teeth > Jaw (bullish alignment), short when price < Lips and Lips < Teeth < Jaw (bearish alignment).
Requires volume > 1.3x 20-period average for confirmation. Uses weekly timeframe for trend filter to reduce whipsaw.
Target: 10-25 trades/year by requiring strong trend alignment and volume confirmation. Works in bull markets via trend following and in bear via short signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA)"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # First value is SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Williams Alligator
    df_wk = get_htf_data(prices, '1w')
    close_wk = df_wk['close'].values
    high_wk = df_wk['high'].values
    low_wk = df_wk['low'].values
    
    # Williams Alligator parameters (13,8,5)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Typical price for Alligator calculation
    typical_price_wk = (high_wk + low_wk + close_wk) / 3.0
    
    # Calculate Smoothed Moving Averages (SMMA)
    jaw_wk = smma(typical_price_wk, jaw_period)
    teeth_wk = smma(typical_price_wk, teeth_period)
    lips_wk = smma(typical_price_wk, lips_period)
    
    # Shift the SMMA values forward by respective periods (as per Williams Alligator)
    jaw_wk = np.roll(jaw_wk, jaw_period//2)
    teeth_wk = np.roll(teeth_wk, teeth_period//2)
    lips_wk = np.roll(lips_wk, lips_period//2)
    
    # Align weekly Alligator lines to daily timeframe
    jaw_wk_aligned = align_htf_to_ltf(prices, df_wk, jaw_wk)
    teeth_wk_aligned = align_htf_to_ltf(prices, df_wk, teeth_wk)
    lips_wk_aligned = align_htf_to_ltf(prices, df_wk, lips_wk)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_wk_aligned[i]) or np.isnan(teeth_wk_aligned[i]) or 
            np.isnan(lips_wk_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > lips AND lips > teeth > jaw (bullish alignment) + volume
            if (close[i] > lips_wk_aligned[i] and 
                lips_wk_aligned[i] > teeth_wk_aligned[i] and 
                teeth_wk_aligned[i] > jaw_wk_aligned[i] and
                volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < lips AND lips < teeth < jaw (bearish alignment) + volume
            elif (close[i] < lips_wk_aligned[i] and 
                  lips_wk_aligned[i] < teeth_wk_aligned[i] and 
                  teeth_wk_aligned[i] < jaw_wk_aligned[i] and
                  volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < lips OR bearish alignment (lips < teeth < jaw)
            if (close[i] < lips_wk_aligned[i] or 
                lips_wk_aligned[i] < teeth_wk_aligned[i] or 
                teeth_wk_aligned[i] < jaw_wk_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > lips OR bullish alignment (lips > teeth > jaw)
            if (close[i] > lips_wk_aligned[i] or 
                lips_wk_aligned[i] > teeth_wk_aligned[i] or 
                teeth_wk_aligned[i] > jaw_wk_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WK1_WilliamsAlligator_Filtered_V1"
timeframe = "1d"
leverage = 1.0