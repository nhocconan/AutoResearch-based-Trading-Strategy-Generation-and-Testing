#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with daily VWAP filter and volume surge confirmation.
# Long when: Alligator lines are bullish (Lips > Teeth > Jaw), price > VWAP, volume > 2x average
# Short when: Alligator lines are bearish (Lips < Teeth < Jaw), price < VWAP, volume > 2x average
# Exit when: Alligator lines cross (Lips crosses Teeth) or price crosses VWAP in opposite direction
# Williams Alligator identifies trend direction and strength, VWAP filters institutional interest, volume confirms strength.
# Target: 20-30 trades/year per symbol. Works in trending markets (both bull and bear) by capturing sustained moves.
name = "4h_WilliamsAlligator_VWAP_VolumeSurge"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Williams Alligator and VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator lines (SMMA = Smoothed Moving Average)
    # Jaw: SMMA of median price, period 13, shift 8 bars
    # Teeth: SMMA of median price, period 8, shift 5 bars  
    # Lips: SMMA of median price, period 5, shift 3 bars
    median_price_1d = (high_1d + low_1d) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Apply shifts (Jaw +8, Teeth +5, Lips +3)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Calculate VWAP for 1-day period
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, out=np.full_like(vwap_numerator, np.nan), where=vwap_denominator!=0)
    
    # Align 1D data to 4H timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 20-period volume average for surge confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        vwap = vwap_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Bullish Alligator (Lips > Teeth > Jaw), price > VWAP, volume surge
            if (lips > teeth and teeth > jaw and 
                price > vwap and vol > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator (Lips < Teeth < Jaw), price < VWAP, volume surge
            elif (lips < teeth and teeth < jaw and 
                  price < vwap and vol > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator crosses bearish OR price crosses below VWAP
            if (lips <= teeth or price < vwap):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator crosses bullish OR price crosses above VWAP
            if (lips >= teeth or price > vwap):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals