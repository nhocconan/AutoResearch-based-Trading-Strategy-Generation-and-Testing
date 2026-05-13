#!/usr/bin/env python3
"""
1d_WilliamsAlligator_1wTrend_Volume
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction on weekly timeframe. Enter long when price > Lips + Teeth > Jaw (bullish alignment) with volume spike; enter short when price < Lips + Teeth < Jaw (bearish alignment) with volume spike. Exit when Alligator alignment breaks or price crosses Jaw. Uses 1d timeframe with 1h Williams Alligator for trend filter to reduce whipsaw. Volume confirmation ensures momentum. Designed for low trade frequency (<15/year) to avoid fee drag in bear markets.
"""

name = "1d_WilliamsAlligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)"""
    # SMMA (Smoothed Moving Average) calculation
    def smma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan, dtype=float)
        result = np.full_like(series, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    median_price = (high + low) / 2.0
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for Williams Alligator trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Williams Alligator on 1h
    jaw_1h, teeth_1h, lips_1h = williams_alligator(high_1h, low_1h, close_1h)
    
    # Align Alligator lines to daily timeframe
    jaw_1h_aligned = align_htf_to_ltf(prices, df_1h, jaw_1h)
    teeth_1h_aligned = align_htf_to_ltf(prices, df_1h, teeth_1h)
    lips_1h_aligned = align_htf_to_ltf(prices, df_1h, lips_1h)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_1h_aligned[i]) or np.isnan(teeth_1h_aligned[i]) or 
            np.isnan(lips_1h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.8x 20-day average
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Bullish Alligator alignment (Lips > Teeth > Jaw) + price above Lips + volume spike
            if (lips_1h_aligned[i] > teeth_1h_aligned[i] > jaw_1h_aligned[i] and 
                close[i] > lips_1h_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment (Lips < Teeth < Jaw) + price below Lips + volume spike
            elif (lips_1h_aligned[i] < teeth_1h_aligned[i] < jaw_1h_aligned[i] and 
                  close[i] < lips_1h_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks bearish OR price crosses below Jaw
            if (lips_1h_aligned[i] < teeth_1h_aligned[i] or 
                close[i] < jaw_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks bullish OR price crosses above Jaw
            if (lips_1h_aligned[i] > teeth_1h_aligned[i] or 
                close[i] > jaw_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals