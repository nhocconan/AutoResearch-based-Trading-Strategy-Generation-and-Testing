#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume spike + chop regime filter
# Uses Williams Alligator (jaw/teeth/lips) for trend direction, Elder Ray (bull/bear power) for momentum
# Volume confirmation (>1.5x 20 EMA) ensures participation, chop regime (<61.8) avoids false signals in ranging markets
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h.
# Works in both bull and bear: Alligator identifies trend, Elder Ray confirms momentum, chop filter avoids whipsaws.

name = "12h_WilliamsAlligator_ElderRay_VolumeChop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA smoothed 8 bars ahead
    # Teeth: 8-period SMMA smoothed 5 bars ahead  
    # Lips: 5-period SMMA smoothed 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Using high for jaw as per original Alligator
    teeth = smma(low, 8)   # Using low for teeth
    lips = smma(close, 5)  # Using close for lips
    
    # Shift to align with proper timing (Alligator uses future smoothing)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray on 12h timeframe using 1d EMA13
    bull_power = high - ema_13_aligned
    bear_power = low - ema_13_aligned
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Choppiness Index regime filter on 12h timeframe
    def choppiness_index(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index"""
        if len(high_arr) < period:
            return np.full_like(close_arr, np.nan, dtype=float)
        atr_sum = np.zeros(len(close_arr))
        true_range = np.maximum(np.abs(high_arr - low_arr),
                               np.maximum(np.abs(high_arr - np.roll(close_arr, 1)),
                                         np.abs(low_arr - np.roll(close_arr, 1))))
        true_range[0] = np.abs(high_arr[0] - low_arr[0])  # First TR
        
        # Calculate ATR (smoothed)
        atr = np.zeros(len(close_arr))
        atr[period-1] = np.mean(true_range[:period])
        for i in range(period, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        
        # Calculate highest high and lowest low over period
        highest_high = np.zeros(len(close_arr))
        lowest_low = np.zeros(len(close_arr))
        for i in range(len(close_arr)):
            if i < period:
                highest_high[i] = np.max(high_arr[:i+1])
                lowest_low[i] = np.min(low_arr[:i+1])
            else:
                highest_high[i] = np.max(high_arr[i-period+1:i+1])
                lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        
        # Choppiness Index = 100 * LOG10(sum(ATR)/period) / LOG10(highest_high - lowest_low)
        chop = np.full_like(close_arr, np.nan, dtype=float)
        for i in range(period-1, len(close_arr)):
            if highest_high[i] > lowest_low[i]:
                chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / period) / np.log10(highest_high[i] - lowest_low[i])
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + Volume spike + Chop < 61.8 (trending)
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                volume[i] > (1.5 * vol_ema_20[i]) and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short conditions: Jaw > Teeth > Lips (bearish alignment) + Bear Power < 0 + Volume spike + Chop < 61.8 (trending)
            elif (jaw[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  volume[i] > (1.5 * vol_ema_20[i]) and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips < Jaw OR Bull Power < 0 OR Chop > 61.8 (ranging)
            if (lips[i] < jaw[i] or 
                bull_power[i] < 0 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Jaw < Lips OR Bear Power > 0 OR Chop > 61.8 (ranging)
            if (jaw[i] < lips[i] or 
                bear_power[i] > 0 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals