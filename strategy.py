#!/usr/bin/env python3
"""
1d_Williams_Alligator_Jaw_Teeth_Lips_Trend
Hypothesis: Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs on smoothed median price) identifies trend direction. 
Go long when Lips > Teeth > Jaw (bullish alignment) + price above Teeth + volume > 1.5x 20-day average.
Go short when Lips < Teeth < Jaw (bearish alignment) + price below Teeth + volume > 1.5x 20-day average.
Exit when alignment breaks or price crosses Teeth. Williams Alligator works in both bull (trend following) 
and bear (catching reversals) markets by filtering noise. Designed for 1d timeframe to limit trades (<25/year) 
and avoid fee drag. Uses weekly timeframe for trend confirmation to reduce whipsaw.
"""

name = "1d_Williams_Alligator_Jaw_Teeth_Lips_Trend"
timeframe = "1d"
leverage = 1.0

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
    
    # Get daily data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator uses smoothed median price: (high + low) / 2
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Calculate SMAs with smoothing (using previous value as in Williams Alligator)
    # Jaw: 13-period SMMA of median price
    jaw = np.full_like(median_price, np.nan)
    teeth = np.full_like(median_price, np.nan)
    lips = np.full_like(median_price, np.nan)
    
    # Smoothed Moving Average (SMMA) calculation
    # SMMA today = (SMMA yesterday * (period-1) + price today) / period
    for i in range(len(median_price)):
        if i == 0:
            jaw[i] = median_price[i]
            teeth[i] = median_price[i]
            lips[i] = median_price[i]
        else:
            if i >= 13:
                jaw[i] = (jaw[i-1] * 12 + median_price[i]) / 13
            else:
                jaw[i] = np.nan
                
            if i >= 8:
                teeth[i] = (teeth[i-1] * 7 + median_price[i]) / 8
            else:
                teeth[i] = np.nan
                
            if i >= 5:
                lips[i] = (lips[i-1] * 4 + median_price[i]) / 5
            else:
                lips[i] = np.nan
    
    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get weekly trend confirmation (price above/below weekly 50 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Bullish alignment (Lips > Teeth > Jaw) + price above Teeth + volume spike + price above weekly EMA50
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > teeth_aligned[i] and vol_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish alignment (Lips < Teeth < Jaw) + price below Teeth + volume spike + price below weekly EMA50
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < teeth_aligned[i] and vol_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alignment breaks or price crosses below Teeth
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alignment breaks or price crosses above Teeth
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals