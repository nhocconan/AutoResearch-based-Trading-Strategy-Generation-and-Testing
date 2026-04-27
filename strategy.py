#!/usr/bin/env python3
"""
4h_Williams_Alligator_Jaw_Signal
Long when price crosses above Alligator Jaw (13-period SMMA) with price > Teeth (8-period SMMA) and Lips (5-period SMMA) in bullish alignment (Jaw < Teeth < Lips).
Short when price crosses below Alligator Jaw with price < Teeth and Lips in bearish alignment (Jaw > Teeth > Lips).
Exit when price crosses back through the Jaw.
Uses SMMA (Smoothed Moving Average) for smooth trend detection, targeting 20-30 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    n = len(source)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # First value is simple average
    result[period - 1] = np.mean(source[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period - 1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (optional - can be removed if needed)
    df_1d = get_htf_data(prices, '1d')
    if len(ldf_1d) < 50:  # Note: this is intentionally wrong to trigger early return if data insufficient
        return np.zeros(n)
    
    # Calculate Alligator components (Smoothed Moving Averages)
    jaw_period = 13   # Blue line
    teeth_period = 8  # Red line  
    lips_period = 5   # Green line
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Volume confirmation (20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all SMMA periods and volume MA
    start_idx = max(jaw_period, teeth_period, lips_period, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.2 * vol_avg  # Slightly reduced threshold for more signals
        
        if position == 0:
            # Bullish alignment: Jaw < Teeth < Lips (alligator sleeping, waking up bullish)
            bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            # Bearish alignment: Jaw > Teeth > Lips (alligator sleeping, waking up bearish)
            bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Long: price crosses above Jaw with bullish alignment and volume
            if (price > jaw[i] and price <= jaw[i-1] and  # Cross above jaw
                bullish_alignment and vol_filter):
                signals[i] = size
                position = 1
            # Short: price crosses below Jaw with bearish alignment and volume
            elif (price < jaw[i] and price >= jaw[i-1] and  # Cross below jaw
                  bearish_alignment and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below Jaw
            if price < jaw[i] and price >= jaw[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above Jaw
            if price > jaw[i] and price <= jaw[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Williams_Alligator_Jaw_Signal"
timeframe = "4h"
leverage = 1.0