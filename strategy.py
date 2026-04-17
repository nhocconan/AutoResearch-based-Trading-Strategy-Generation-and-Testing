#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray combination with volume confirmation.
Long when Alligator jaws (13) < teeth (8) < lips (5) AND Bull Power > 0 AND volume > 1.5x 20-period average.
Short when Alligator jaws > teeth > lips AND Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when Alligator lines re-cross (jaws-teeth-lips convergence) or volume drops below average.
Uses proven Williams Alligator trend identification with Elder Ray power measurement.
Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
"""

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
    
    # Get 12h data for indicator calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Williams Alligator (SMMA = smoothed moving average)
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator lines: Jaw (13, 8), Teeth (8, 5), Lips (5, 3)
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_12h - ema13_12h
    bear_power = low_12h - ema13_12h
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Alligator aligned (JAW < TEETH < LIPS) AND Bull Power > 0 AND volume > 1.5x avg
            if jaw_val < teeth_val < lips_val and bull > 0 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (JAW > TEETH > LIPS) AND Bear Power < 0 AND volume > 1.5x avg
            elif jaw_val > teeth_val > lips_val and bear < 0 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator lines converge (JAW >= TEETH or TEETH >= LIPS) OR volume < average
            if jaw_val >= teeth_val or teeth_val >= lips_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator lines converge (JAW <= TEETH or TEETH <= LIPS) OR volume < average
            if jaw_val <= teeth_val or teeth_val <= lips_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0