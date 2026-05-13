#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Trend_With_Volume
Hypothesis: Williams Alligator (13,8,5 SMAs with 8,5,3 offsets) identifies trends.
In strong trends (JAW > TEETH > LIPS for long, JAW < TEETH < LIPS for short),
enter with volume confirmation (volume > 1.5x 20-period average).
Exit when Alligator lines re-interlace (trend weakness) or volume drops.
Designed for low trade frequency (~15-25/year) on 12h to minimize fee drag.
"""

name = "12h_WilliamsAlligator_Trend_With_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Williams Alligator (calculated on 1d close)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: three SMAs with different periods and offsets
    # Jaw: 13-period SMMA, offset 8 bars
    # Teeth: 8-period SMMA, offset 5 bars  
    # Lips: 5-period SMMA, offset 3 bars
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Apply offsets: shift values to the right (past values)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Set NaN for rolled values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align to 12t - use previous day's values (available at 12h open)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any Alligator value is NaN (not enough data)
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: JAW > TEETH > LIPS (bullish alignment) + volume confirmation
            if (jaw_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: JAW < TEETH < LIPS (bearish alignment) + volume confirmation
            elif (jaw_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator lines re-interlace (trend weakness) OR volume drops
            if not (jaw_aligned[i] > teeth_aligned[i] and 
                    teeth_aligned[i] > lips_aligned[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator lines re-interlace OR volume drops
            if not (jaw_aligned[i] < teeth_aligned[i] and 
                    teeth_aligned[i] < lips_aligned[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals