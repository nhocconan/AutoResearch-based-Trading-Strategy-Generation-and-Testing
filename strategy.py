#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) with 
volume > 1.5x 20-day average AND 1w close > 1w open (bullish weekly candle).
Short when jaws cross below teeth with volume confirmation AND 1w close < 1w open (bearish weekly candle).
Exit when jaws re-cross teeth in opposite direction.
Williams Alligator uses smoothed moving averages (SMMA) which reduce whipsaw in ranging markets.
Designed to catch trends in both bull and bear markets with weekly confirmation to avoid counter-trend trades.
Target: 10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (Williams Alligator)"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is simple SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Calculate 1w bullish/bearish candle
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator SMMA on 1d close
    jaw = smma(close_1d, 13)   # Jaw (13-period SMMA)
    teeth = smma(close_1d, 8)  # Teeth (8-period SMMA)
    lips = smma(close_1d, 5)   # Lips (5-period SMMA) - not used in signals but part of Alligator
    
    # Calculate 1d volume MA (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (1d)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Williams Alligator crossover signals
        jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
        jaw_below_teeth = jaw_aligned[i] < teeth_aligned[i]
        
        # Previous bar values for crossover detection
        if i > 0:
            jaw_above_teeth_prev = jaw_aligned[i-1] > teeth_aligned[i-1]
            jaw_below_teeth_prev = jaw_aligned[i-1] < teeth_aligned[i-1]
        else:
            jaw_above_teeth_prev = False
            jaw_below_teeth_prev = False
        
        # Bullish crossover: jaw crosses above teeth
        bullish_crossover = jaw_above_teeth and not jaw_above_teeth_prev
        # Bearish crossover: jaw crosses below teeth
        bearish_crossover = jaw_below_teeth and not jaw_below_teeth_prev
        
        if position == 0:
            # Long: bullish crossover with volume confirmation and weekly bullish candle
            if (bullish_crossover and volume_confirmed and weekly_bullish_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish crossover with volume confirmation and weekly bearish candle
            elif (bearish_crossover and volume_confirmed and weekly_bearish_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish crossover (jaw crosses below teeth)
            if bearish_crossover:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish crossover (jaw crosses above teeth)
            if bullish_crossover:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0