#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 12-hour Williams Alligator with 1-day volume confirmation and 1-week trend filter
# Designed for low trade frequency (target 12-37/year) with clear trend following logic
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends
# Works in both bull (teeth above jaw) and bear (teeth below jaw) markets
# Uses volume spike and weekly trend filter to avoid false signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator components on 12h
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    
    def smma(series, period):
        """Smoothed Moving Average"""
        if len(series) < period:
            return np.full_like(series, np.nan)
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        result = np.full_like(series, np.nan)
        result[period-1] = sma[period-1]
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Smooth the lines
    jaw_smoothed = smma(jaw, 8)
    teeth_smoothed = smma(teeth, 5)
    lips_smoothed = smma(lips, 3)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (close price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_smoothed)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_smoothed)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_smoothed)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(close_1w_aligned[i])):
            continue
        
        # Williams Alligator conditions:
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Long entry: Bullish alignment + price above teeth + volume spike + weekly uptrend
        if (bullish_alignment and 
            close[i] > teeth_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            close[i] > close_1w_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bearish alignment + price below teeth + volume spike + weekly downtrend
        elif (bearish_alignment and 
              close[i] < teeth_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              close[i] < close_1w_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite alignment or price crosses jaw
        elif position == 1 and ((lips_aligned[i] < teeth_aligned[i]) or 
                                close[i] < jaw_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and ((lips_aligned[i] > teeth_aligned[i]) or 
                                 close[i] > jaw_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0