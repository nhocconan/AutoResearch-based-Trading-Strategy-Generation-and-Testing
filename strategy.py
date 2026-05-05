#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Long when price > Alligator Jaw AND 1d EMA50 uptrend AND volume > 1.5x 20-period average
# Short when price < Alligator Jaw AND 1d EMA50 downtrend AND volume > 1.5x 20-period average
# Exit when price crosses Alligator Teeth or trend reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-30 trades/year per symbol.
# Alligator identifies trend, EMA50 filters higher timeframe direction, volume confirms strength.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# 12h timeframe minimizes fee drag while capturing medium-term Alligator signals.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
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
    
    # Get 12h data ONCE before loop for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    # Jaw (Blue): 13-period SMMA smoothed 8 bars ahead
    # Teeth (Red): 8-period SMMA smoothed 5 bars ahead
    # Lips (Green): 5-period SMMA smoothed 3 bars ahead
    # We'll use Jaw as the main trend indicator (slowest)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_12h = df_12h['close'].values
    jaw = smma(close_12h, 13)  # Jaw (Blue line)
    teeth = smma(close_12h, 8)  # Teeth (Red line)
    lips = smma(close_12h, 5)   # Lips (Green line)
    
    # Align Alligator Jaw to prices timeframe (main trend indicator)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.ones(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Alligator Jaw AND 1d EMA50 uptrend AND volume confirmation
            if (close[i] > jaw_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Alligator Jaw AND 1d EMA50 downtrend AND volume confirmation
            elif (close[i] < jaw_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Alligator Teeth OR 1d trend changes to downtrend
            # Use teeth for tighter exit (faster line)
            teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
            if (np.isnan(teeth_aligned[i]) or 
                close[i] < teeth_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Alligator Teeth OR 1d trend changes to uptrend
            teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
            if (np.isnan(teeth_aligned[i]) or 
                close[i] > teeth_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals