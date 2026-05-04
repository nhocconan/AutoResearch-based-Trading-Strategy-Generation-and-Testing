#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d regime filter
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via SMAs with future shift
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# Long when: Alligator aligned bullish (JAW > TEETH > LIPS) AND Bull Power > 0 AND 1d EMA50 uptrend
# Short when: Alligator aligned bearish (JAW < TEETH < LIPS) AND Bear Power > 0 AND 1d EMA50 downtrend
# Uses 6h for entry timing with 1d HTF for regime filter to reduce whipsaw
# Target: 12-37 trades/year (50-150 total over 4 years) with discrete sizing 0.25
# Works in bull markets via longs in uptrend and bear via shorts in downtrend

name = "6h_Alligator_ElderRay_1dEMA50_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for HTF regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for regime filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: SMAs with future shift
    # JAW: 13-period SMMA shifted 8 bars
    # TEETH: 8-period SMMA shifted 5 bars  
    # LIPS: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply Alligator shifts (JAW+8, TEETH+5, LIPS+3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for shifted positions that would look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Elder Ray Power on 6h
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish AND Bull Power > 0 AND 1d uptrend
            if (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i] and
                bull_power[i] > 0 and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish AND Bear Power > 0 AND 1d downtrend
            elif (jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i] and
                  bear_power[i] > 0 and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power <= 0 OR 1d trend turns down
            if (not (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]) or
                bull_power[i] <= 0 or
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power <= 0 OR 1d trend turns up
            if (not (jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]) or
                bear_power[i] <= 0 or
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals