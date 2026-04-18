#!/usr/bin/env python3
"""
12h Williams Alligator with Volume Spike and EMA Trend Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trends; 
price above/below all three lines indicates strong trend. 
Volume spike confirms institutional participation. 
EMA50 on daily timeframe filters for higher-timeframe trend alignment.
Works in both bull and bear markets by following Alligator alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(arr, period):
    """Simple Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period-1, len(arr)):
        result[i] = np.mean(arr[i-period+1:i+1])
    return result

def smma(arr, period):
    """Smoothed Moving Average (used in Williams Alligator)"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    # First value is SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA(i) = (SMMA(i-1) * (period-1) + close(i)) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator on daily: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # All are SMMA (Smoothed Moving Average) with different shifts
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = smma(median_price.values, 13)  # Blue line
    jaw = np.roll(jaw, 8)  # Shifted by 8 bars forward
    
    teeth = smma(median_price.values, 8)  # Red line
    teeth = np.roll(teeth, 5)  # Shifted by 5 bars forward
    
    lips = smma(median_price.values, 5)  # Green line
    lips = np.roll(lips, 3)  # Shifted by 3 bars forward
    
    # EMA50 on daily for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price above all AND EMA50 uptrend AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                close[i] > lips_aligned[i] and
                close[i] > ema_50_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price below all AND EMA50 downtrend AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  close[i] < lips_aligned[i] and
                  close[i] < ema_50_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator lines cross (Lips < Teeth) OR price below Teeth
            if lips_aligned[i] < teeth_aligned[i] or close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator lines cross (Lips > Teeth) OR price above Teeth
            if lips_aligned[i] > teeth_aligned[i] or close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_VolumeSpike_EMA50Trend"
timeframe = "12h"
leverage = 1.0