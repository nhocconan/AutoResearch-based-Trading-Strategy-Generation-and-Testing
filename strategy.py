#!/usr/bin/env python3
"""
12-hour Williams Alligator with 1-day Trend Filter and Volume Confirmation.
Long when price > Alligator Teeth + 1-day uptrend + volume spike.
Short when price < Alligator Teeth + 1-day downtrend + volume spike.
Exit when price crosses Alligator Jaw or trend reverses.
Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends: price above teeth = uptrend.
Designed for low trade frequency by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 1-day trend.
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
    
    # Williams Alligator components
    # Jaw (blue): 13-period SMMA smoothed 8 bars ahead
    # Teeth (red): 8-period SMMA smoothed 5 bars ahead  
    # Lips (green): 5-period SMMA smoothed 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for rolled values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price > Teeth + 1-day uptrend + volume spike
            if close[i] > teeth[i] and ema20_1d_aligned[i] > ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < Teeth + 1-day downtrend + volume spike
            elif close[i] < teeth[i] and ema20_1d_aligned[i] < ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Jaw or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price < Jaw or 1-day trend turns down
                if close[i] < jaw[i] or ema20_1d_aligned[i] < ema20_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price > Jaw or 1-day trend turns up
                if close[i] > jaw[i] or ema20_1d_aligned[i] > ema20_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0