#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator breakout with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) from 6h for trend direction and entry signals
# 1d EMA50 as higher timeframe trend filter to reduce whipsaw in ranging markets
# Volume confirmation (>1.3x 20-bar average) ensures breakout authenticity
# Designed for 6h timeframe to capture medium-term swings in both bull and bear markets
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from end
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate volume confirmation (>1.3x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)  # Will use previous completed 1d bar
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND uptrend (price > EMA50) AND volume
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > lips_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Jaw > Teeth > Lips (bearish alignment) AND price < Lips AND downtrend (price < EMA50) AND volume
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < lips_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment (Jaw > Teeth > Lips) OR price crosses below Teeth
            if (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] or 
                close[i] < teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment (Lips > Teeth > Jaw) OR price crosses above Teeth
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] or 
                close[i] > teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals