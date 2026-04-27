#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using Williams Alligator (jaw, teeth, lips) from 1-day timeframe with volume confirmation.
# The Alligator uses smoothed moving averages to identify trends: when the three lines are intertwined (sleeping),
# the market is ranging; when they diverge and align in order (awake eating), a trend is present.
# Long when price > lips > teeth > jaw (bullish alignment) with volume confirmation.
# Short when price < lips < teeth < jaw (bearish alignment) with volume confirmation.
# Uses 1-day Williams Alligator (13,8,5 SMMA) for trend direction and 12h volume spike (>1.5x 20-period average) for confirmation.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (trend following).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (Smoothed Moving Average - SMMA)
    # Jaw: SMMA(13, 8)
    # Teeth: SMMA(8, 5)
    # Lips: SMMA(5, 3)
    def smma(source, period):
        sma = np.full_like(source, np.nan, dtype=float)
        if len(source) < period:
            return sma
        # First value is simple SMA
        sma[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            sma[i] = (sma[i-1] * (period-1) + source[i]) / period
        return sma
    
    jaw_1d = smma(close_1d, 13)
    teeth_1d = smma(close_1d, 8)
    lips_1d = smma(close_1d, 5)
    
    # Align Alligator lines to 12h timeframe (wait for 1d bar to close)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: price > lips > teeth > jaw
        bullish_alignment = (close[i] > lips_1d_aligned[i] and 
                           lips_1d_aligned[i] > teeth_1d_aligned[i] and 
                           teeth_1d_aligned[i] > jaw_1d_aligned[i])
        
        # Bearish alignment: price < lips < teeth < jaw
        bearish_alignment = (close[i] < lips_1d_aligned[i] and 
                           lips_1d_aligned[i] < teeth_1d_aligned[i] and 
                           teeth_1d_aligned[i] < jaw_1d_aligned[i])
        
        # Enter long on bullish alignment with volume
        if bullish_alignment and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Enter short on bearish alignment with volume
        elif bearish_alignment and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions: when alignment breaks
        elif position == 1 and not bullish_alignment:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not bearish_alignment:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeFilter"
timeframe = "12h"
leverage = 1.0