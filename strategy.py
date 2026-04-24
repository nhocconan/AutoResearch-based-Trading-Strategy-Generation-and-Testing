#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray combination with volume confirmation and chop regime filter.
- Primary timeframe: 12h for lower trade frequency and better signal quality.
- HTF: 1d for Williams Alligator (JAW/TEETH/LIPS) and Elder Ray (Bull/Bear Power).
- Williams Alligator: JAW=SMMA(13,8), TEETH=SMMA(8,5), LIPS=SMMA(5,3). Trend when LIPS > TEETH > JAW (bull) or LIPS < TEETH < JAW (bear).
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13). Confirms trend strength.
- Volume confirmation: current volume > 1.5 * 20-period volume MA.
- Chop regime filter: Choppiness Index(14) < 61.8 to avoid ranging markets.
- Entry: Long when Alligator bullish AND Bull Power > 0 AND volume spike AND chop < 61.8.
         Short when Alligator bearish AND Bear Power < 0 AND volume spike AND chop < 61.8.
- Exit: When Alligator reverses (LIPS crosses TEETH) or opposite signal appears.
- Works in bull via buying strong uptrends, in bear via selling strong downtrends.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) aka Wilder's Moving Average"""
    if length < 1:
        return source
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is simple SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def calculate_choppiness(high, low, close, length=14):
    """Choppiness Index: higher = ranging, lower = trending"""
    if len(high) < length * 2:
        return np.full_like(high, np.nan)
    
    atr = np.zeros_like(high)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # ATR calculation using Wilder's smoothing
    atr[length-1] = np.mean(tr[:length])
    for i in range(length, len(tr)):
        atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
    
    # Sum of ATR over length period
    atr_sum = np.zeros_like(high)
    for i in range(length-1, len(high)):
        atr_sum[i] = np.sum(atr[i-length+1:i+1])
    
    # Highest high and lowest low over length period
    hh = np.zeros_like(high)
    ll = np.zeros_like(low)
    for i in range(length-1, len(high)):
        hh[i] = np.max(high[i-length+1:i+1])
        ll[i] = np.min(low[i-length+1:i+1])
    
    # Choppiness Index formula
    chop = np.zeros_like(high)
    for i in range(length-1, len(high)):
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(length)
        else:
            chop[i] = 50  # Neutral when range is zero
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need enough for SMMA(13) and EMA(13)
        return np.zeros(n)
    
    # Calculate Williams Alligator components (SMMA)
    jaw = smma(df_1d['close'].values, 13)  # SMMA(13,8)
    teeth = smma(df_1d['close'].values, 8)  # SMMA(8,5)
    lips = smma(df_1d['close'].values, 5)   # SMMA(5,3)
    
    # Calculate Elder Ray components
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Calculate Choppiness Index for regime filter
    chop = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align 1d indicators to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 20, 14)  # Need enough for Alligator, EMA, volume MA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator trend conditions
        alligator_bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and regime filter
            if volume_spike[i] and chop_aligned[i] < 61.8:  # Avoid ranging markets
                # Bullish entry: Alligator bullish AND Bull Power positive
                if alligator_bullish and bull_power_aligned[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Alligator bearish AND Bear Power negative
                elif alligator_bearish and bear_power_aligned[i] < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator reverses (lips crosses below teeth) or opposite signal
            if lips_aligned[i] < teeth_aligned[i]:  # Alligator turning bearish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator reverses (lips crosses above teeth) or opposite signal
            if lips_aligned[i] > teeth_aligned[i]:  # Alligator turning bullish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0