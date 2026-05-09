#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX(14) + Williams Alligator (5,3,4) + volume confirmation
# Uses ADX to identify trending markets (>25) and Alligator to determine direction
# Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
# Long: Price > Teeth > Jaw + ADX > 25 + volume > 1.5x average
# Short: Price < Teeth < Jaw + ADX > 25 + volume > 1.5x average
# Exit: ADX < 20 (trend weakening) or price crosses Jaw in opposite direction
# Designed for 6h timeframe with target of 50-150 trades over 4 years (12-37/year)
# Works in both bull and bear by requiring strong trend confirmation (ADX > 25)

name = "6h_ADX_Alligator_Volume"
timeframe = "6h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev*(period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX components (14-period)
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
        low_diff = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
            
        tr[i] = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values: prev*(1-1/period) + current*(1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    plus_di = 100 * wilders_smooth(plus_dm, 14) / wilders_smooth(tr, 14)
    minus_di = 100 * wilders_smooth(minus_dm, 14) / wilders_smooth(tr, 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Williams Alligator (5,3,4) - Smoothed Moving Averages
    jaw = smma(df_1d['close'].values, 13)  # Jaw (13-period)
    teeth = smma(df_1d['close'].values, 8)  # Teeth (8-period)
    lips = smma(df_1d['close'].values, 5)   # Lips (5-period)
    
    # Align indicators to 6h
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX and Alligator calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_6h[i]) or np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or 
            np.isnan(lips_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from Alligator
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_6h[i] > teeth_6h[i] and teeth_6h[i] > jaw_6h[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips_6h[i] < teeth_6h[i] and teeth_6h[i] < jaw_6h[i]
        
        # Price position relative to Teeth
        price_above_teeth = close[i] > teeth_6h[i]
        price_below_teeth = close[i] < teeth_6h[i]
        
        # ADX trend strength
        strong_trend = adx_6h[i] > 25
        weak_trend = adx_6h[i] < 20
        
        if position == 0:
            # Enter long: bullish alignment + price above teeth + strong trend + volume
            if bullish_alignment and price_above_teeth and strong_trend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + price below teeth + strong trend + volume
            elif bearish_alignment and price_below_teeth and strong_trend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment OR weak trend OR price crosses below jaw
            if bearish_alignment or weak_trend or close[i] < jaw_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment OR weak trend OR price crosses above jaw
            if bullish_alignment or weak_trend or close[i] > jaw_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals