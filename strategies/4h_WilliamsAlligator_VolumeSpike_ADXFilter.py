#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + ADX Trend Filter
Uses Williams Alligator (3 SMAs: Jaw, Teeth, Lips) to detect trend direction.
Long when price above all three lines with volume spike, short when price below.
ADX filter ensures we only trade in trending markets (ADX > 25).
Designed for low trade frequency with strong trend-following edge.
"""

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
    
    # Williams Alligator components (13, 8, 5 period SMAs with future shifts)
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        # Convert SMA to SMMA: first value is SMA, then SMMA = (prev*(period-1) + current)/period
        smma_vals = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(arr)):
                if not np.isnan(smma_vals[i-1]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
        return smma_vals
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply Alligator shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # ADX calculation for trend strength
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smm(arr, period):
            """Smoothed Moving Average for ADX"""
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) >= period:
                # First value is sum of first 'period' elements
                result[period-1] = np.nansum(arr[:period])
                for i in range(period, len(arr)):
                    if not np.isnan(result[i-1]):
                        result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        tr_smooth = smm(tr, period)
        plus_dm_smooth = smm(plus_dm, period)
        minus_dm_smooth = smm(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smm(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_above = lips[i] > teeth[i] and teeth[i] > jaw[i]
        lips_below = lips[i] < teeth[i] and teeth[i] < jaw[i]
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: price above all Alligator lines with volume spike and strong trend
            if (price > lips[i] and price > teeth[i] and price > jaw[i] and
                lips_above and volume_spike[i] and strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price below all Alligator lines with volume spike and strong trend
            elif (price < lips[i] and price < teeth[i] and price < jaw[i] and
                  lips_below and volume_spike[i] and strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses below Teeth (Alligator "sleeping" signal)
            if price < teeth[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses above Teeth (Alligator "sleeping" signal)
            if price > teeth[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_WilliamsAlligator_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0