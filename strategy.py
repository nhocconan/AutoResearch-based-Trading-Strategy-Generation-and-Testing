#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with volume confirmation
# Uses 1d Williams Alligator (Jaw/Teeth/Lips) for trend direction and 6h Elder Ray (Bull/Bear power) for momentum
# Enters only when Bull Power > 0 and Bear Power < 0 with volume confirmation
# Targets 50-150 total trades over 4 years (12-37/year) with strict entry conditions
# Works in bull/bear by following 1d Alligator trend and using Elder Ray for entry timing
name = "6h_1dWilliamsAlligator_ElderRay_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # Jaw (Blue)
    teeth = smma(close_1d, 8)  # Teeth (Red)
    lips = smma(close_1d, 5)   # Lips (Green)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 6d data for Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df_6d = get_htf_data(prices, '6d')
    close_6d = df_6d['close'].values
    # Calculate EMA13 on 6d close
    ema_13_6d = pd.Series(close_6d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_6d_aligned = align_htf_to_ltf(prices, df_6d, ema_13_6d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6d_aligned
    bear_power = low - ema_13_6d_aligned
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend based on Alligator alignment
        # Bullish trend: Lips > Teeth > Jaw
        # Bearish trend: Jaw > Teeth > Lips
        bullish_trend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_trend = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: Bullish trend AND Bull Power > 0 AND Bear Power < 0 with volume
            if (bullish_trend and 
                bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish trend AND Bull Power < 0 AND Bear Power > 0 with volume
            elif (bearish_trend and 
                  bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend turns bearish or Elder Ray signals weaken
            if not bullish_trend or bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend turns bullish or Elder Ray signals weaken
            if not bearish_trend or bull_power[i] >= 0 or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals