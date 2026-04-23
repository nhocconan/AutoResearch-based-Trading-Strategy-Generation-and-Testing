#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + Chop Regime Filter
Long when Alligator jaws < teeth < lips (bullish alignment) + volume > 2x avg + chop > 61.8 (range)
Short when Alligator jaws > teeth > lips (bearish alignment) + volume > 2x avg + chop > 61.8
Exit when alignment breaks or chop < 38.2 (trend)
Designed for choppy markets (2025) with low trade frequency to avoid fee drag.
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
    
    # Williams Alligator (13,8,5) smoothed with SMMA
    jaw_len, teeth_len, lips_len = 13, 8, 5
    jaw_offset, teeth_offset, lips_offset = 8, 5, 3
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(smma(close, jaw_len), jaw_offset)
    teeth = smma(smma(close, teeth_len), teeth_offset)
    lips = smma(smma(close, lips_len), lips_offset)
    
    # Chopiness Index (14-period)
    def chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = smma(tr, period)
        
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        hh_ll = highest_high - lowest_low
        
        chop_val = np.full_like(close, np.nan, dtype=float)
        for i in range(len(close)):
            if atr[i] > 0 and hh_ll[i] > 0:
                chop_val[i] = 100 * np.log10(atr[i] * period / hh_ll[i]) / np.log10(period)
        return chop_val
    
    chop_val = chop(high, low, close, 14)
    
    # Average volume for spike detection
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(chop_val[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish_align = jaw[i] < teeth[i] < lips[i]
        bearish_align = jaw[i] > teeth[i] > lips[i]
        chop_high = chop_val[i] > 61.8  # ranging market
        chop_low = chop_val[i] < 38.2   # trending market
        volume_spike = volume[i] > 2.0 * avg_volume[i]
        
        if position == 0:
            # Enter long/short only in ranging markets with volume spike
            if bullish_align and chop_high and volume_spike:
                signals[i] = 0.25
                position = 1
            elif bearish_align and chop_high and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: alignment breaks or market trends
                if not bullish_align or chop_low:
                    exit_signal = True
            else:  # position == -1
                # Exit short: alignment breaks or market trends
                if not bearish_align or chop_low:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_Chop_VolumeSpike"
timeframe = "4h"
leverage = 1.0