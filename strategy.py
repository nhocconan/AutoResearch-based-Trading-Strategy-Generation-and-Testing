#!/usr/bin/env python3
"""
6h ADX + Williams Alligator with Volume Confirmation
Long: ADX>25 + Alligator bullish (jaw<teeth<lips) + volume > 1.5x avg volume
Short: ADX>25 + Alligator bearish (jaw>teeth>lips) + volume > 1.5x avg volume
Exit: Opposite Alligator alignment or ADX<20
Uses ADX for trend strength, Alligator for direction, volume for confirmation.
Designed to capture strong trends in both bull and bear markets while avoiding chop.
Target: 80-160 total trades over 4 years (20-40/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines"""
    # Jaw (Blue): 13-period SMMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean()
    # Teeth (Red): 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean()
    # Lips (Green): 5-period SMMA shifted 3 bars
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean()
    return jaw, teeth, lips

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(high) - pd.Series(high).shift(1)
    dm_minus = pd.Series(low).shift(1) - pd.Series(low)
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=period, min_periods=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period, min_periods=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h
    jaw, teeth, lips = calculate_alligator(close)
    
    # ADX on 6h
    adx = calculate_adx(high, low, close)
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need ADX and Alligator calculations
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        if position == 0:
            # Long: ADX>25 + Alligator bullish + volume confirmation
            if (adx[i] > 25 and 
                jaw[i] < teeth[i] and teeth[i] < lips[i] and  # jaw < teeth < lips (bullish)
                vol > 1.5 * avg_vol):
                signals[i] = 0.25
                position = 1
            # Short: ADX>25 + Alligator bearish + volume confirmation
            elif (adx[i] > 25 and 
                  jaw[i] > teeth[i] and teeth[i] > lips[i] and  # jaw > teeth > lips (bearish)
                  vol > 1.5 * avg_vol):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Opposite Alligator alignment OR ADX<20 (weak trend)
            if (jaw[i] > teeth[i] or teeth[i] > lips[i]) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Opposite Alligator alignment OR ADX<20
            if (jaw[i] < teeth[i] or teeth[i] < lips[i]) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Alligator_Volume"
timeframe = "6h"
leverage = 1.0