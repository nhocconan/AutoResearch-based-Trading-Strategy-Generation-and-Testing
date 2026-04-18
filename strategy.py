#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three SMAs
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (red): 8-period SMMA, shifted 5 bars forward  
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = np.full_like(series, np.nan)
        if len(series) >= period:
            sma[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    # Calculate SMMA for each line
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply forward shifts (Williams Alligator uses future-shifted averages)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    # Jaw: shift 8 bars forward
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    # Teeth: shift 5 bars forward  
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    # Lips: shift 3 bars forward
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Get 6-hour data for entry timing
    # Price close above/below Alligator lines indicates trend direction
    
    # Align Alligator lines to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_align = lips_6h[i] > teeth_6h[i] and teeth_6h[i] > jaw_6h[i]
        bearish_align = lips_6h[i] < teeth_6h[i] and teeth_6h[i] < jaw_6h[i]
        
        if position == 0:
            # Long: price above all lines with bullish alignment and volume
            if close[i] > lips_6h[i] and close[i] > teeth_6h[i] and close[i] > jaw_6h[i] and bullish_align and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below all lines with bearish alignment and volume
            elif close[i] < lips_6h[i] and close[i] < teeth_6h[i] and close[i] < jaw_6h[i] and bearish_align and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Teeth (8-period) or Alligator loses alignment
            if close[i] < teeth_6h[i] or not (lips_6h[i] > teeth_6h[i] and teeth_6h[i] > jaw_6h[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Teeth (8-period) or Alligator loses alignment
            if close[i] > teeth_6h[i] or not (lips_6h[i] < teeth_6h[i] and teeth_6h[i] < jaw_6h[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_Trend_Volume"
timeframe = "6h"
leverage = 1.0