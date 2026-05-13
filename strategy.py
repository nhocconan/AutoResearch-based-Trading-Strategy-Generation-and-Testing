#!/usr/bin/env python3
"""
1d_WilliamsAlligator_JawTeeth_Cross
Hypothesis: Williams Alligator (Jaw=13, Teeth=8, Lips=5) crossover signals trend changes. 
Go long when Teeth crosses above Jaw with confirmation (close > Lips), short when Teeth crosses below Jaw with confirmation (close < Lips).
Filtered by weekly trend (price above/below weekly EMA50) and volume spike (volume > 1.5x 20-day average).
Williams Alligator works in both bull (catching trend starts) and bear (catching trend reversals) markets.
Designed for 1d timeframe to limit trades (<20/year) and avoid fee drag.
"""

name = "1d_WilliamsAlligator_JawTeeth_Cross"
timeframe = "1d"
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
    
    # Get daily data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA
    # SMMA calculation: smoothed moving average
    def smma(series, period):
        sma = np.zeros(len(series))
        sma[:period] = np.nan
        if len(series) >= period:
            sma[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw = smma(close_1d, 13)  # Jaw (blue line)
    teeth = smma(close_1d, 8)  # Teeth (red line)
    lips = smma(close_1d, 5)   # Lips (green line)
    
    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Teeth crosses above Jaw (bullish) + volume spike + close above Lips + price above weekly EMA50
            if (teeth_aligned[i-1] <= jaw_aligned[i-1] and teeth_aligned[i] > jaw_aligned[i] and 
                vol_spike and close[i] > lips_aligned[i] and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Teeth crosses below Jaw (bearish) + volume spike + close below Lips + price below weekly EMA50
            elif (teeth_aligned[i-1] >= jaw_aligned[i-1] and teeth_aligned[i] < jaw_aligned[i] and 
                  vol_spike and close[i] < lips_aligned[i] and close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Teeth crosses below Jaw or close below Lips
            if teeth_aligned[i] < jaw_aligned[i] or close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Teeth crosses above Jaw or close above Lips
            if teeth_aligned[i] > jaw_aligned[i] or close[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals